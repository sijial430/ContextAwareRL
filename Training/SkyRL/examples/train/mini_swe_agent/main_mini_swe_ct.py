"""Choose-Trajectory training entry point for Mini-SWE-Agent.

Mirrors ``main_mini_swe.py`` but adds an auxiliary DPO-style loss on a
side "choose-trajectory" dataset (single-turn MCQ prompts) while the RL
path continues to roll out the agent against the SWE-Bench sandbox.

Run via the accompanying ``run_mini_swe_agentforge8B_ct.sh`` script.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import ray
from loguru import logger

from skyrl.train.config.config import (
    AlgorithmConfig,
    DataConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)
from skyrl.train.entrypoints.main_base import validate_cfg
from skyrl.train.utils import initialize_ray

from .ct_dataset import ChooseTrajectoryDataset
from .ct_trainer import ChooseTrajectoryRayPPOTrainer
from .main_mini_swe import MiniSWEPPOExp
from .mini_swe_generator import MiniSWEGeneratorConfig


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CTDataConfig(DataConfig):
    choose_trajectory_data: List[str] = field(default_factory=list)
    """Paths (jsonl) to single-turn A/B MCQ prompts used for the CT aux loss."""
    choose_trajectory_batch_size: int = 12
    """Number of CT samples per optimizer step. Must be divisible by policy dp_size."""
    choose_trajectory_max_prompt_length: int = 32768
    """Max token length for a CT prompt (after chat template + prefill).
    Rows that tokenize longer than this are filtered out."""


@dataclass
class CTAlgorithmConfig(AlgorithmConfig):
    choose_trajectory_coef: float = 0.005
    """Weight of the DPO-sigmoid loss relative to the PPO policy loss."""
    choose_trajectory_clip: float = 5.0
    """Clip range C for the log ratio before logsigmoid (``clamp(-C, C)``)."""
    choose_trajectory_prefill: str = "The correct option is"
    """Text appended to the chat template output to anchor A/B prediction."""
    choose_trajectory_enable_thinking: bool = False
    """``enable_thinking`` flag passed to ``tokenizer.apply_chat_template``
    (Qwen3-family models). Must match the inference-time setting used in
    ``eval_klear_prefill_logits.py``."""


@dataclass
class CTTrainerConfig(TrainerConfig):
    algorithm: CTAlgorithmConfig = field(default_factory=CTAlgorithmConfig)


@dataclass
class CTSkyRLTrainConfig(SkyRLTrainConfig):
    data: CTDataConfig = field(default_factory=CTDataConfig)
    trainer: CTTrainerConfig = field(default_factory=CTTrainerConfig)
    generator: MiniSWEGeneratorConfig = field(default_factory=MiniSWEGeneratorConfig)


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class MiniSWECTPPOExp(MiniSWEPPOExp):
    def __init__(self, cfg: CTSkyRLTrainConfig):
        super().__init__(cfg)
        self.choose_trajectory_dataset: Optional[ChooseTrajectoryDataset] = (
            self._build_choose_trajectory_dataset()
        )

    def _build_choose_trajectory_dataset(self) -> Optional[ChooseTrajectoryDataset]:
        ct_paths = self.cfg.data.choose_trajectory_data
        if not ct_paths:
            logger.info("[CT] no data.choose_trajectory_data provided; CT loss disabled")
            return None
        logger.info(f"[CT] building dataset from: {ct_paths}")
        ct_dataset = ChooseTrajectoryDataset(
            datasets=list(ct_paths),
            tokenizer=self.tokenizer,
            prefill=self.cfg.trainer.algorithm.choose_trajectory_prefill,
            max_prompt_length=self.cfg.data.choose_trajectory_max_prompt_length,
            enable_thinking=self.cfg.trainer.algorithm.choose_trajectory_enable_thinking,
        )
        if len(ct_dataset) == 0:
            logger.warning("[CT] dataset is empty after filtering; CT loss will be disabled")
            return None
        return ct_dataset

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return ChooseTrajectoryRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
            choose_trajectory_dataset=self.choose_trajectory_dataset,
        )

    def _setup_trainer(self):
        """Replicate :meth:`BasePPOExp._setup_trainer` but inject the
        CT-aware :class:`CTPolicyWorker` instead of the stock FSDP worker.
        Everything else (critic worker, ref worker, ray placement) is
        untouched.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy not in ("fsdp", "fsdp2"):
            raise ValueError(
                f"Choose-trajectory training only supports fsdp strategies, "
                f"got {self.cfg.trainer.strategy}"
            )

        from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
            CriticWorker,
            RefWorker,
        )

        from .ct_policy_worker import CTPolicyWorker as PolicyWorker

        tracker = self.get_tracker()
        inference_engine_client = self.get_inference_client()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)
        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker)
        return trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: CTSkyRLTrainConfig):
    exp = MiniSWECTPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = CTSkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
