"""
FastV-inspired visual token pruner.

Reference: "An Image is Worth 1/2 Tokens After Layer 2"
           Liang Chen et al., ECCV 2024  (arXiv 2403.06764)

Mechanism:
  1. Run LLM layers 0..K (causal mask, position_ids, use_cache=False)
     with output_attentions=True at layer K
  2. importance[v] = mean attention received by visual token v
                     averaged over ALL query positions and ALL heads
     (paper criterion: column-mean of attention matrix at layer K)
  3. Keep top keep_ratio visual tokens; prune the rest from original f_embs
  4. Feed pruned f_embs to generate() → full N-layer inference with fewer tokens

Note on implementation vs paper:
  - Paper: single forward pass, prune mid-inference at layer K, continue K+1..N
  - This implementation: K-layer pre-pass for importance, then full generate with pruned input
  - Token SELECTION is logically equivalent; timing overhead differs
  - Requires attn_implementation="eager" (FlashAttention does not expose attn weights)

Settings: prune_layer=8, keep_ratio=0.5  →  256 → 128 visual tokens
"""
import time
from typing import Optional, Tuple

import torch


def _make_causal_mask(
    lm_model,
    f_embs: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """
    Returns 4D additive causal mask [1, 1, seq_len, seq_len].
    Tries the model's own _update_causal_mask first (version-safe),
    falls back to manual construction if unavailable.
    """
    # 1순위: 모델 내부 메서드 (transformers 버전별 포맷 차이 없이 안전)
    if hasattr(lm_model, "_update_causal_mask"):
        try:
            cache_position = torch.arange(seq_len, dtype=torch.long, device=device)
            mask = lm_model._update_causal_mask(
                attention_mask=None,
                input_tensor=f_embs,
                cache_position=cache_position,
                past_key_values=None,
                output_attentions=False,
            )
            if mask is not None:
                return mask
        except Exception:
            pass  # fallthrough to manual

    # 2순위: 표준 additive causal mask (0=attend, -inf=block)
    mask = torch.full(
        (1, 1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.triu(diagonal=1)   # upper-tri = -inf, lower+diag = 0


class FastVPruner:
    """
    Args:
        keep_ratio  : fraction of visual tokens to keep (0.5 → 256→128)
        prune_layer : LLM layer index at which to extract attention (0-indexed)
    """

    def __init__(self, keep_ratio: float = 0.5, prune_layer: int = 8):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio  = keep_ratio
        self.prune_layer = prune_layer

    @torch.no_grad()
    def prune(
        self,
        lm_model,               # model.language_model.model  (Qwen3Model)
        f_embs: torch.Tensor,   # [1, seq_len, D_llm]  full fused input
        f_mask: torch.Tensor,   # [1, seq_len]
        vis_start: int,         # token index where visual tokens begin
        n_visual: int,          # number of visual tokens  (256)
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            pruned_f_embs : [1, seq_len - n_pruned, D_llm]
            pruned_f_mask : [1, seq_len - n_pruned]
            t_ms          : wall-clock pruning time in milliseconds
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        seq_len = f_embs.shape[1]
        vis_end = vis_start + n_visual
        dtype   = f_embs.dtype

        # ── 4D causal mask ────────────────────────────────────────────────────
        # 모델 자체의 _update_causal_mask 우선 사용 (버전 호환성 보장)
        # 없으면 표준 additive causal mask 수동 구성
        causal_mask = _make_causal_mask(lm_model, f_embs, seq_len, dtype, device)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # ── Run layers 0 .. prune_layer ───────────────────────────────────────
        hidden: torch.Tensor        = f_embs
        attn_at_K: Optional[torch.Tensor] = None

        for i, layer in enumerate(lm_model.layers[: self.prune_layer + 1]):
            is_K = (i == self.prune_layer)
            out  = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=is_K,   # weights only needed at layer K
                use_cache=False,
            )
            hidden = out[0]               # [1, seq_len, D]
            if is_K and len(out) > 1:
                attn_at_K = out[1]        # [1, n_heads, seq_len, seq_len]

        # ── Compute visual token importance (paper criterion) ────────────────
        # FastV paper: importance[v] = mean attention received by visual token v
        #              averaged over ALL query positions and ALL heads.
        #
        # Causal attention ensures:
        #   - pre-visual positions (0..vis_start-1) → visual: softmax ≈ 0 (future blocked)
        #   - visual positions (vis_start..vis_end-1) → earlier visual: non-zero (causal)
        #   - post-visual text (vis_end..seq_len-1)  → visual: non-zero (past visible)
        #
        # Using all positions (not just text) matches the paper and preserves
        # spatial importance signals from visual self-attention.
        n_keep = max(1, int(n_visual * self.keep_ratio))

        if attn_at_K is not None:
            # all_to_vis: [n_heads, seq_len, n_visual]
            all_to_vis = attn_at_K[0, :, :, vis_start:vis_end]
            importance = all_to_vis.mean(dim=(0, 1))   # [n_visual]
        else:
            # Fallback: L2 norm of visual hidden states at layer K
            importance = hidden[0, vis_start:vis_end, :].norm(dim=-1)

        keep_idx = importance.topk(n_keep).indices.sort().values

        # ── Prune original f_embs (not layer-K hidden states) ─────────────────
        # We feed the pruned *original* embeddings to generate(), not the
        # intermediate hidden states, so the full model runs cleanly from scratch.
        pruned_f_embs = torch.cat([
            f_embs[:, :vis_start, :],
            f_embs[:, vis_start:vis_end, :][:, keep_idx, :],
            f_embs[:, vis_end:, :],
        ], dim=1)   # [1, vis_start + n_keep + n_post, D]

        pruned_f_mask = torch.cat([
            f_mask[:, :vis_start],
            f_mask[:, vis_start:vis_end][:, keep_idx],
            f_mask[:, vis_end:],
        ], dim=1)

        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        return pruned_f_embs, pruned_f_mask, t_ms
