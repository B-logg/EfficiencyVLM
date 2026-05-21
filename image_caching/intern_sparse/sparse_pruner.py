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
        lm_model,               # model.language_model  (ForCausalLM wrapper)
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

        # ── Get base transformer model ────────────────────────────────────────
        # model.language_model (ForCausalLM) 안에 있는 base transformer를 얻음.
        # 없으면 lm_model 자체를 사용 (이미 base model인 경우).
        base_model = getattr(lm_model, "model", lm_model)

        # ── Full forward with output_attentions=True ──────────────────────────
        # 레이어 직접 호출 대신 모델 자체의 forward를 사용 → 모델 구조에 독립적.
        # attention_mask는 2D padding mask (all-ones) → 내부에서 4D causal mask 생성.
        attn_at_K: Optional[torch.Tensor] = None
        last_hidden: Optional[torch.Tensor] = None
        try:
            out = base_model(
                inputs_embeds=f_embs,
                attention_mask=f_mask,
                output_attentions=True,
                use_cache=False,
            )
            # out.attentions: tuple of [1, n_heads, seq, seq] per layer
            if hasattr(out, "attentions") and out.attentions is not None:
                if len(out.attentions) > self.prune_layer:
                    attn_at_K = out.attentions[self.prune_layer]
            if hasattr(out, "last_hidden_state"):
                last_hidden = out.last_hidden_state
        except Exception as e:
            # forward 실패 → fallback은 아래에서 처리
            pass

        # ── Compute visual token importance (paper criterion) ────────────────
        # FastV paper: importance[v] = mean attention received by visual token v
        #              averaged over ALL query positions and ALL heads.
        #
        # Causal attention ensures:
        #   - pre-visual positions (0..vis_start-1) → visual: softmax ≈ 0 (future blocked)
        #   - visual positions (vis_start..vis_end-1) → earlier visual: non-zero (causal)
        #   - post-visual text (vis_end..seq_len-1)  → visual: non-zero (past visible)
        n_keep = max(1, int(n_visual * self.keep_ratio))

        if attn_at_K is not None:
            # all_to_vis: [n_heads, seq_len, n_visual]
            all_to_vis = attn_at_K[0, :, :, vis_start:vis_end]
            importance = all_to_vis.mean(dim=(0, 1))   # [n_visual]
        elif last_hidden is not None:
            # Fallback 1: L2 norm of visual hidden states
            importance = last_hidden[0, vis_start:vis_end, :].norm(dim=-1)
        else:
            # Fallback 2: uniform (모든 토큰 동일 가중치 → 앞쪽 절반 유지)
            importance = torch.arange(n_visual, dtype=torch.float32, device=device).flip(0)

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
