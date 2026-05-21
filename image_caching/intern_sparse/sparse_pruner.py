"""
True single-pass FastV visual token pruner for InternVL3.5 / Qwen3.

Reference: "An Image is Worth 1/2 Tokens After Layer 2"
           Liang Chen et al., ECCV 2024  (arXiv 2403.06764)

Mechanism
---------
  1. Prefill  : single forward pass (ALL N layers), use_cache=True,
                output_attentions=True  → builds full KV cache
  2. Importance: column-mean of attention at layer K over all heads
                and all query positions (causal rows for pre-visual
                positions are ≈ 0 and do not contaminate the signal)
  3. Prune KV  : remove unimportant visual token positions from every
                layer's key/value cache (no extra forward needed)
  4. First tok : compute first generated token directly from the prefill
                logits at the final sequence position
  5. Decode    : generate() resumes with the pruned KV cache
                → every decode step touches a shorter sequence

Comparison with paper
---------------------
  Paper : prune mid-prefill at layer K → layers K+1..N during prefill
          already see the shorter sequence
  Here  : prune KV cache AFTER full prefill → layers K+1..N during
          prefill see all tokens (one-time cost, not repeated per step)
  Token selection    : identical criterion
  Decode phase       : identical (same pruned KV cache)
  Timing difference  : paper saves compute in prefill layers K+1..N;
                       our approach eliminates the separate 2-pass overhead

Settings: prune_layer=8, keep_ratio=0.5  →  256 → 128 visual tokens
"""
import time
from typing import Tuple

import torch

try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False


def _prune_kv_cache(past_kvs, keep_seq_idx: torch.Tensor):
    """
    Remove pruned positions from every layer of the KV cache.

    Handles both DynamicCache (transformers ≥ 4.38) and the legacy
    tuple-of-tuples format.
    """
    if _HAS_DYNAMIC_CACHE and isinstance(past_kvs, DynamicCache):
        new_cache = DynamicCache()
        new_cache.key_cache   = [k[:, :, keep_seq_idx, :] for k in past_kvs.key_cache]
        new_cache.value_cache = [v[:, :, keep_seq_idx, :] for v in past_kvs.value_cache]
        n_kept = int(keep_seq_idx.shape[0])
        # Attribute name changed between transformers versions
        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(new_cache, attr):
                setattr(new_cache, attr, n_kept)
                break
        return new_cache
    else:
        # Legacy tuple-of-tuples: ((k0, v0), (k1, v1), ...)
        return tuple(
            (kv[0][:, :, keep_seq_idx, :], kv[1][:, :, keep_seq_idx, :])
            for kv in past_kvs
        )


class FastVPruner:
    """
    Args:
        keep_ratio  : fraction of visual tokens to keep (0.5 → 256→128)
        prune_layer : LLM layer index for attention extraction (0-indexed)
    """

    def __init__(self, keep_ratio: float = 0.5, prune_layer: int = 8):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio  = keep_ratio
        self.prune_layer = prune_layer

    @torch.no_grad()
    def prefill_and_prune(
        self,
        lm_model,               # model.language_model  (ForCausalLM wrapper)
        f_embs: torch.Tensor,   # [1, seq_len, D_llm]
        f_mask: torch.Tensor,   # [1, seq_len]
        vis_start: int,         # index where visual tokens begin
        n_visual: int,          # number of visual tokens (256)
        device: str,
    ) -> Tuple[torch.Tensor, object, torch.Tensor, float]:
        """
        Single prefill → prune KV cache → return first generated token.

        Returns:
            first_token_id  : [1, 1]       token ID of first generated token
            pruned_past_kvs : KV cache with unimportant visual positions removed
            pruned_f_mask   : [1, new_seq]  attention mask for the pruned sequence
            t_ms            : wall-clock time in milliseconds
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        seq_len = f_embs.shape[1]
        vis_end = vis_start + n_visual
        n_keep  = max(1, int(n_visual * self.keep_ratio))

        base_model = getattr(lm_model, "model", lm_model)

        # ── Single prefill: all layers, all tokens ─────────────────────────────
        # output_attentions=True  : get attention weights at every layer
        # use_cache=True          : build KV cache for the subsequent decode phase
        out = base_model(
            inputs_embeds=f_embs,
            attention_mask=f_mask,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        # ── Visual token importance (FastV paper criterion) ────────────────────
        # Column-mean of attention at layer K, averaged over all heads and all
        # query positions → importance[v] = how much token v is attended to.
        # Pre-visual positions have causal-masked rows (≈ 0 for visual columns)
        # so they do not corrupt the signal; post-visual text rows dominate.
        attn_K     = out.attentions[self.prune_layer]        # [1, H, seq, seq]
        to_vis     = attn_K[0, :, :, vis_start:vis_end]      # [H, seq, n_visual]
        importance = to_vis.mean(dim=(0, 1))                  # [n_visual]

        keep_local = importance.topk(n_keep).indices.sort().values   # indices within vis

        # Full-sequence positions to retain
        keep_seq = torch.cat([
            torch.arange(vis_start, device=device),
            keep_local + vis_start,
            torch.arange(vis_end, seq_len, device=device),
        ])

        # ── Prune KV cache for ALL N layers ───────────────────────────────────
        pruned_kvs  = _prune_kv_cache(out.past_key_values, keep_seq)

        # ── Prune attention mask ───────────────────────────────────────────────
        pruned_mask = torch.cat([
            f_mask[:, :vis_start],
            f_mask[:, vis_start:vis_end][:, keep_local],
            f_mask[:, vis_end:],
        ], dim=1)   # [1, vis_start + n_keep + n_post]

        # ── First generated token from prefill logits ──────────────────────────
        # base_model.forward() already applies the final LayerNorm before
        # returning last_hidden_state, so we pass it directly to lm_head.
        logits      = lm_model.lm_head(out.last_hidden_state[:, -1:, :])  # [1,1,V]
        first_token = logits.argmax(dim=-1)                                # [1, 1]

        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        return first_token, pruned_kvs, pruned_mask, t_ms
