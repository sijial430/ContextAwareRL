"""Choose-Trajectory-aware RayPPOTrainer.

Holds a side dataloader of single-turn MCQ prompts and, for every policy
optimizer step, fetches one CT mini-batch, stages it per DP rank, and
passes it as a kwarg to each actor's ``forward_backward`` alongside the RL
chunk. Falls back to the base trainer when no CT dataset is attached, so
the same file is safe to run without CT data.

Everything else (critic path, checkpointing, validation, reward) delegates
to :class:`RayPPOTrainer` unchanged.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional

import ray
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.backends.skyrl_train.distributed.dispatch import MeshDispatch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics
from skyrl.train.trainer import RayPPOTrainer

from .ct_dataset import ChooseTrajectoryDataset


class ChooseTrajectoryRayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        *args,
        choose_trajectory_dataset: Optional[ChooseTrajectoryDataset] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.choose_trajectory_dataset = choose_trajectory_dataset
        self.choose_trajectory_dataloader: Optional[StatefulDataLoader] = None
        self._ct_iter = None
        # CT dataloader construction is deferred to _init_ct_dataloader()
        # because self.policy_model is None until build_models() is called.
        self._ct_initialized = False

    def _init_ct_dataloader(self):
        """Build the CT dataloader lazily, after ``build_models()`` has set
        ``self.policy_model`` so we can read dp_size."""
        if self._ct_initialized:
            return
        self._ct_initialized = True

        if self.choose_trajectory_dataset is None or len(self.choose_trajectory_dataset) == 0:
            logger.info("[CT] no CT dataset provided; falling back to vanilla policy updates")
            return

        ct_batch_size = self.cfg.data.choose_trajectory_batch_size
        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        if ct_batch_size % dp_size != 0:
            raise ValueError(
                f"choose_trajectory_batch_size ({ct_batch_size}) must be "
                f"divisible by policy dp_size ({dp_size})"
            )

        self.choose_trajectory_dataloader = StatefulDataLoader(
            dataset=self.choose_trajectory_dataset,
            batch_size=ct_batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self.choose_trajectory_dataset.collate_fn,
            drop_last=True,
            pin_memory=False,
        )
        self._ct_iter = iter(self.choose_trajectory_dataloader)
        logger.info(
            f"[CT] dataloader ready: size={len(self.choose_trajectory_dataloader)} "
            f"batch_size={ct_batch_size} dp_size={dp_size}"
        )

    def _next_ct_batch(self) -> Optional[TrainingInputBatch]:
        if self.choose_trajectory_dataloader is None:
            return None
        try:
            return next(self._ct_iter)
        except StopIteration:
            self._ct_iter = iter(self.choose_trajectory_dataloader)
            return next(self._ct_iter)

    def _execute_training_step(self, model: str, data: TrainingInputBatch) -> Dict[str, float]:
        # Lazy-init the CT dataloader on the first training step (after
        # build_models has populated self.policy_model).
        if not self._ct_initialized:
            self._init_ct_dataloader()

        # Only augment the policy path with CT; critic path stays vanilla.
        if model != "policy" or self.choose_trajectory_dataloader is None:
            return super()._execute_training_step(model, data)

        n_samples = self.cfg.generator.n_samples_per_prompt
        mini_batch_size = self.cfg.trainer.policy_mini_batch_size * n_samples
        all_metrics: Dict[str, List[float]] = defaultdict(list)

        all_chunk_refs = self.dispatch.stage_data(model, data, mini_batch_size)

        actor_group = self.dispatch._actor_groups[model]
        actor_infos = actor_group.actor_infos
        dp_size = actor_infos[0].rank.dp_size

        for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
            for chunk_refs in all_chunk_refs:
                ct_batch = self._next_ct_batch()
                # One CT mini-batch per optimizer step (mirroring
                # EasyR1_Modify's inner ``ci_micro_batches`` loop).
                ct_chunk_nested = MeshDispatch.stage_chunks(
                    dp_size=dp_size,
                    data=ct_batch,
                    mini_batch_size=len(ct_batch),
                )
                ct_chunk_refs = ct_chunk_nested[0]  # (dp_size,) ObjectRefs

                # Replicate the GPU-onload guarantees that
                # ``forward_backward_from_staged`` would give us if we were
                # going through dispatch — but we need per-actor-info
                # arguments, so we dispatch manually.
                self.dispatch._ensure_on_gpu(model, need_optimizer=True, need_model=True)

                object_refs = []
                for actor_info in actor_infos:
                    rl_ref = chunk_refs[actor_info.rank.dp]
                    ct_ref = ct_chunk_refs[actor_info.rank.dp]
                    object_refs.append(
                        actor_info.handle.forward_backward.remote(
                            rl_ref,
                            choose_trajectory_chunk=ct_ref,
                        )
                    )
                statuses = ray.get(object_refs)
                status = statuses[0]

                # Match the memory-snapshot hook that the vanilla path uses.
                self.dispatch._save_memory_snapshot(model, "forward_backward")

                for k, v in status.items():
                    all_metrics[k].append(v)

                grad_norm = self.dispatch.optim_step(model)
                if grad_norm is not None:
                    all_metrics["grad_norm"].append(grad_norm)

        all_metrics.pop("loss_fn_outputs", None)
        reduced: Dict[str, Any] = reduce_metrics(all_metrics)
        return reduced

    # -- checkpoint resume hooks: save/load CT dataloader state along with the
    # rest of the trainer state so resumption doesn't silently restart the CT
    # stream from scratch. Purely best-effort; failure is logged not raised.

    def save_checkpoints(self):
        super().save_checkpoints()
        if self.choose_trajectory_dataloader is None:
            return
        import os

        import torch

        from skyrl.backends.skyrl_train.utils.io import io as skyrl_io

        global_step_folder = os.path.join(
            self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}"
        )
        ct_path = os.path.join(global_step_folder, "ct_dataloader.pt")
        try:
            with skyrl_io.open_file(ct_path, "wb") as f:
                torch.save(self.choose_trajectory_dataloader.state_dict(), f)
            logger.info(f"[CT] saved dataloader state to {ct_path}")
        except Exception as e:
            logger.warning(f"[CT] failed to save dataloader state: {e}")

    def load_checkpoints(self):
        result = super().load_checkpoints()
        if self.choose_trajectory_dataloader is None:
            return result
        import os

        import torch

        from skyrl.backends.skyrl_train.utils.io import io as skyrl_io

        global_step, checkpoint_path = result
        if checkpoint_path is None:
            return result
        ct_path = os.path.join(checkpoint_path, "ct_dataloader.pt")
        if not skyrl_io.exists(ct_path):
            logger.info(f"[CT] no saved dataloader state at {ct_path}; starting fresh")
            return result
        try:
            with skyrl_io.open_file(ct_path, "rb") as f:
                state = torch.load(f, map_location="cpu", weights_only=False)
            self.choose_trajectory_dataloader.load_state_dict(state)
            self._ct_iter = iter(self.choose_trajectory_dataloader)
            logger.info(f"[CT] restored dataloader state from {ct_path}")
        except Exception as e:
            logger.warning(f"[CT] failed to restore dataloader state: {e}; starting fresh")
        return result
