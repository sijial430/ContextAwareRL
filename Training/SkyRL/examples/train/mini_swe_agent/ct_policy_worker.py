"""Choose-Trajectory FSDP PolicyWorker.

Extends the stock :class:`FSDPPolicyWorkerBase` with a DPO-style auxiliary
loss computed on a single-turn MCQ prompt ("choose trajectory"). Mirrors the
EasyR1_Modify ``dp_actor.update_policy`` logic but adapted for SkyRL's
micro-batch gradient-accumulation model where ``optim_step`` scales all
accumulated grads by ``1 / n_rl_micro``.

Math (per optimizer step):

* RL gradient contribution (unchanged)::

    sum_i grad(rl_loss_i) / n_rl_micro  ==  mean_i grad(rl_loss_i)

* CT gradient contribution (newly added, scaled to stay commensurable)::

    sum_j grad(coef * ct_loss_j * n_rl_micro / n_ct_micro) / n_rl_micro
      == coef * sum_j grad(ct_loss_j) / n_ct_micro
      == coef * mean_j grad(ct_loss_j)

  Critically, CT micro-batches do **not** increment
  ``_micro_batches_accumulated``, so the RL path's denominator is not
  polluted and the RL effective learning rate is unchanged.

Sequence parallel: the CT forward temporarily clears the global ulysses
group so the monkey-patched ``_flash_attention_forward`` takes its
``ulysses_sp_size == 1`` branch and runs vanilla flash attention on the
full (unsliced) prompt. This keeps the CT implementation simple at the
cost of some compute on the SP ranks; good enough for a first cut per the
plan that the user approved.
"""

import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import ray
import torch
import torch.nn.functional as F
from loguru import logger

from skyrl.backends.skyrl_train.distributed.ulysses.utils import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import (
    BatchIterator,
    all_reduce_metrics,
    reduce_metrics,
)

_CT_TOPK = 5


@contextmanager
def _disable_ulysses_group():
    """Temporarily unset the ulysses sp group so the monkey-patched
    flash-attention forward falls back to its ``sp_size == 1`` branch.

    Safe because :func:`get_ulysses_sequence_parallel_world_size` returns 1
    when the group is ``None``, which skips all the all-to-all logic in
    :func:`_ulysses_flash_attention_forward`.
    """
    old_group = get_ulysses_sequence_parallel_group()
    if old_group is not None:
        set_ulysses_sequence_parallel_group(None)
    try:
        yield
    finally:
        if old_group is not None:
            set_ulysses_sequence_parallel_group(old_group)


class CTFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    def _ct_get_debug_tokenizer(self):
        """Lazy-load a tokenizer on the worker for top-k debug decoding.

        The worker already has the policy model loaded via FSDP, but not the
        tokenizer. Loading it here is cheap (tokenizer-only, no weights) and
        happens at most once per worker process.
        """
        tok = getattr(self, "_ct_debug_tokenizer", None)
        if tok is not None:
            return tok
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                self.cfg.policy.model.path, trust_remote_code=True
            )
        except Exception as e:  # pragma: no cover — debug path only
            logger.warning(f"[CT] failed to load debug tokenizer: {e}; top-k decode disabled")
            tok = None
        self._ct_debug_tokenizer = tok
        return tok

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
        choose_trajectory_chunk: Optional[TrainingInputBatch] = None,
    ) -> Dict[str, float]:
        micro_batch_size = self.cfg.micro_train_batch_size_per_gpu
        all_metrics: Dict[str, list] = defaultdict(list)
        all_loss_fn_outputs = []

        # Count RL micro-batches before iterating so we can scale the CT loss.
        n_rl_micro = max(1, math.ceil(data.batch_size / micro_batch_size))

        for micro_batch in BatchIterator(data, micro_batch_size, drop_last=False):
            metrics = self._forward_backward_micro(
                micro_batch, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            )
            self._micro_batches_accumulated += 1

            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            for k, v in metrics.items():
                all_metrics[k].append(v)

        ct_coef = float(getattr(self.cfg.algorithm, "choose_trajectory_coef", 0.0) or 0.0)
        if choose_trajectory_chunk is not None and ct_coef > 0.0:
            ct_clip = float(getattr(self.cfg.algorithm, "choose_trajectory_clip", 5.0))
            ct_metrics_agg = self._run_choose_trajectory(
                choose_trajectory_chunk,
                n_rl_micro=n_rl_micro,
                coef=ct_coef,
                clip=ct_clip,
                micro_batch_size=micro_batch_size,
            )
            for k, v in ct_metrics_agg.items():
                all_metrics[k].append(v)

        result = reduce_metrics(dict(all_metrics))
        if all_loss_fn_outputs:
            result["loss_fn_outputs"] = all_loss_fn_outputs
        return result

    def _run_choose_trajectory(
        self,
        ct_chunk: TrainingInputBatch,
        n_rl_micro: int,
        coef: float,
        clip: float,
        micro_batch_size: int,
    ) -> Dict[str, float]:
        """Loop over CT micro-batches, backward the scaled DPO loss, return
        a dict of *per-call* metric means (``reduce_metrics`` will average
        across calls later)."""
        # Reclaim GPU memory fragmented by the RL micro-batch loop before
        # running the CT forward (which doesn't benefit from sample packing
        # or ulysses sequence parallelism and thus has a bigger peak footprint).
        torch.cuda.empty_cache()

        self.model.train()
        ct_chunks = list(ct_chunk.chunk(micro_batch_size))
        n_ct_micro = max(1, len(ct_chunks))

        a_ids = [int(i) for i in ct_chunk.metadata["a_ids"]]
        b_ids = [int(i) for i in ct_chunk.metadata["b_ids"]]

        acc: Dict[str, list] = defaultdict(list)
        for ct_mb in ct_chunks:
            metrics = self._forward_backward_ct_micro(
                ct_mb,
                n_rl_micro=n_rl_micro,
                n_ct_micro=n_ct_micro,
                coef=coef,
                clip=clip,
                a_ids=a_ids,
                b_ids=b_ids,
            )
            for k, v in metrics.items():
                acc[k].append(v)

        out: Dict[str, float] = {k: sum(v) / len(v) for k, v in acc.items()}
        out["choose_trajectory/coef"] = coef
        out["choose_trajectory/n_ct_micro"] = float(n_ct_micro)
        out["choose_trajectory/n_rl_micro"] = float(n_rl_micro)
        return out

    def _forward_backward_ct_micro(
        self,
        ct_mb: TrainingInputBatch,
        n_rl_micro: int,
        n_ct_micro: int,
        coef: float,
        clip: float,
        a_ids: List[int],
        b_ids: List[int],
    ) -> Dict[str, float]:
        device = torch.cuda.current_device()
        input_ids = ct_mb["input_ids"].to(device)
        attention_mask = ct_mb["attention_mask"].to(device)
        position_ids = ct_mb["position_ids"].to(device)
        correct_is_a = ct_mb["ct_correct_is_a"].to(device).bool()

        # Disable ulysses for the ENTIRE forward+backward cycle.  Gradient
        # checkpointing recomputes the forward during backward — if ulysses
        # is restored between forward and backward, the recomputed tensor
        # shapes (halved heads) won't match the saved shapes (full heads)
        # and torch raises CheckpointError.
        with _disable_ulysses_group():
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                last_logits = self._forward_prompt_last_logits(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )  # (B, V)
                log_probs_vocab = F.log_softmax(last_logits.float(), dim=-1)
                a_ids_t = torch.as_tensor(a_ids, device=device, dtype=torch.long)
                b_ids_t = torch.as_tensor(b_ids, device=device, dtype=torch.long)
                # Per-sample max log-prob across candidate token ids (e.g.
                # " A" vs "A"). Gradient flows through the argmax candidate.
                log_p_a = log_probs_vocab.index_select(-1, a_ids_t).max(dim=-1).values  # (B,)
                log_p_b = log_probs_vocab.index_select(-1, b_ids_t).max(dim=-1).values  # (B,)
                log_p_correct = torch.where(correct_is_a, log_p_a, log_p_b)
                log_p_wrong = torch.where(correct_is_a, log_p_b, log_p_a)
                r = log_p_correct - log_p_wrong
                r_clipped = r.clamp(-clip, clip)
                ct_loss = -F.logsigmoid(r_clipped).mean()

            # --- Snapshot metrics + top-k debug BEFORE backward. ---
            with torch.no_grad():
                log_probs_vocab_d = log_probs_vocab.detach()
                log_p_a_d = log_p_a.detach()
                log_p_b_d = log_p_b.detach()
                log_p_correct_d = log_p_correct.detach()
                r_d = r.detach()
                r_clipped_d = r_clipped.detach()
                ct_loss_d = ct_loss.detach()

                prob_a_full = log_p_a_d.exp()  # (B,)
                prob_b_full = log_p_b_d.exp()  # (B,)
                prob_correct_ab = torch.sigmoid(r_d)  # (B,)
                prob_correct_full = log_p_correct_d.exp()  # (B,)

                a_ranks = (log_probs_vocab_d > log_p_a_d.unsqueeze(-1)).sum(dim=-1) + 1  # (B,)
                b_ranks = (log_probs_vocab_d > log_p_b_d.unsqueeze(-1)).sum(dim=-1) + 1  # (B,)

                topk = torch.topk(log_probs_vocab_d, k=_CT_TOPK, dim=-1)  # (B, k)
                topk_ids = topk.indices.cpu().tolist()
                topk_lps = topk.values.cpu().tolist()

                metric_values = {
                    "choose_trajectory/loss": float(ct_loss_d.item()),
                    "choose_trajectory/log_ratio": float(r_clipped_d.mean().item()),
                    "choose_trajectory/log_ratio_raw": float(r_d.mean().item()),
                    "choose_trajectory/acc": float((r_d > 0).float().mean().item()),
                    "choose_trajectory/hard_frac": float((r_d < 0).float().mean().item()),
                    "choose_trajectory/prob_correct_ab": float(prob_correct_ab.mean().item()),
                    "choose_trajectory/prob_a_full": float(prob_a_full.mean().item()),
                    "choose_trajectory/prob_b_full": float(prob_b_full.mean().item()),
                    "choose_trajectory/prob_correct_full": float(prob_correct_full.mean().item()),
                    "choose_trajectory/a_rank_mean": float(a_ranks.float().mean().item()),
                    "choose_trajectory/b_rank_mean": float(b_ranks.float().mean().item()),
                }

                debug_lines = None
                if self._rank == 0:
                    tok = self._ct_get_debug_tokenizer()
                    batch_sz = input_ids.size(0)
                    debug_lines = []
                    for i in range(batch_sz):
                        gt = "A" if bool(correct_is_a[i].item()) else "B"
                        parsed = "A" if log_p_a_d[i].item() >= log_p_b_d[i].item() else "B"
                        mark = "OK" if parsed == gt else "NO"
                        la = float(log_p_a_d[i].item())
                        lb = float(log_p_b_d[i].item())
                        pc_ab = float(prob_correct_ab[i].item())
                        ar = int(a_ranks[i].item())
                        br = int(b_ranks[i].item())
                        if tok is not None:
                            top_strs = [
                                f"{tok.decode([int(tid)])!r}={float(lp):+.2f}"
                                for tid, lp in zip(topk_ids[i], topk_lps[i])
                            ]
                        else:
                            top_strs = [
                                f"id={int(tid)}={float(lp):+.2f}"
                                for tid, lp in zip(topk_ids[i], topk_lps[i])
                            ]
                        debug_lines.append(
                            f"[CT:rank0 sample {i}] gt={gt} parsed={parsed} {mark} "
                            f"logP(A)={la:+.3f} logP(B)={lb:+.3f} gap={la - lb:+.3f} "
                            f"P(correct|A,B)={pc_ab:.4f} A#{ar} B#{br} "
                            f"top{_CT_TOPK}=[{', '.join(top_strs)}]"
                        )

            # Pre-scale so the optimizer step's (1 / n_rl_micro) cancels out
            # and the final CT contribution is exactly ``coef * mean(ct_loss)``.
            scale = coef * (n_rl_micro / max(n_ct_micro, 1))
            scaled = ct_loss * scale
            self.strategy.backward(scaled, self.model, self.optimizer)

        # Now that backward has run (and ulysses is restored), emit debug.
        if debug_lines is not None:
            for line in debug_lines:
                logger.info(line)

        ct_metrics = all_reduce_metrics(metric_values, self.strategy)
        return ct_metrics

    def _forward_prompt_last_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward on prompt-only input; return logits at the last non-pad
        token position, one per sample.

        ``self.model`` is an :class:`HFModelWrapper`; ``self.model.model`` is
        the underlying ``AutoModelForCausalLM`` (possibly FSDP-wrapped and
        monkey-patched for ulysses). We call it directly rather than going
        through :meth:`HFModelWrapper.forward`, which expects a
        sequences+num_actions signature geared toward PPO-style response
        log-prob extraction.
        """
        # The caller (_forward_backward_ct_micro) already holds the
        # _disable_ulysses_group() context for the full forward+backward cycle.
        #
        # ``logits_to_keep=1`` tells the model to apply the LM head only at
        # the last sequence position, producing a (B, 1, V) logits tensor
        # instead of the full (B, L, V). For a 16k-token prompt with vocab
        # 152k this cuts the logits tensor from ~4.6 GB to ~0.3 MB and avoids
        # the GPU OOM that killed the worker in earlier runs.
        #
        # Input is left-padded (ct_dataset.collate_fn), so position -1 is
        # always the last non-pad token — the end of the prefill string (or
        # the end of the chat template itself when prefill is empty). The
        # logits at this position predict the next token (A/B variants).
        hf_model = self.model.model
        fwd_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        # MoE: model_wrapper.py sets ``config.output_router_logits=True`` globally
        # so HF computes load_balancing_loss_func and returns per-layer
        # router_logits tensors. CT only reads ``output.logits`` at the last
        # position, so request False here to skip both the aux-loss compute and
        # the router_logits allocation (~800 MB / 32k-prompt for Qwen3-Coder-30B).
        # Guarded on the config field so dense models, which don't accept the
        # kwarg, are unaffected.
        if "output_router_logits" in hf_model.config.to_dict():
            fwd_kwargs["output_router_logits"] = False
        output = hf_model(**fwd_kwargs)
        return output.logits[:, -1, :]  # (B, V)


CTPolicyWorker = ray.remote(num_gpus=1)(CTFSDPPolicyWorkerBase)
