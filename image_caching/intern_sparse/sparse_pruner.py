"""
True FastV-style visual token pruner.

Mechanism (FastV paper: Zheng et al., 2024):
  1. Run LLM layers 0..K with output_attentions=True at layer K
  2. importance[v] = mean over heads and text positions of attn[text → visual[v]]
  3. Keep top keep_ratio visual tokens; discard the rest
  4. Return pruned f_embs for full generate()

Key difference from norm-based pruning:
  - Uses text→visual cross-attention (query-guided, context-aware)
  - Pruning happens AFTER fusion (text tokens needed for attention computation)
  - Requires attn_implementation="eager" (FlashAttention does not return weights)

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

        # ── 4D causal mask  [1, 1, seq_len, seq_len] ─────────────────────────
        # Convention (added to raw attn scores before softmax):
        #   0    = "can attend" (current token + past tokens)
        #   -inf = "masked"    (future tokens)
        # triu(diagonal=1): upper-triangle = -inf, lower+diagonal = 0
        causal_mask = torch.full(
            (1, 1, seq_len, seq_len),
            torch.finfo(dtype).min,
            dtype=dtype, device=device,
        )
        causal_mask = causal_mask.triu(diagonal=1)

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

        # ── Compute visual token importance ───────────────────────────────────
        # Prompt layout: "User: " [vis_tokens] "\nquestion\n...\nAssistant:"
        # Text tokens AFTER visual (indices vis_end .. seq_len-1) attend to visual
        # via causal attention (they appear later in the sequence).
        n_keep = max(1, int(n_visual * self.keep_ratio))

        if attn_at_K is not None and vis_end < seq_len:
            # text_to_vis: [n_heads, n_text_post, n_visual]
            text_to_vis = attn_at_K[0, :, vis_end:, vis_start:vis_end]
            importance  = text_to_vis.mean(dim=(0, 1))   # [n_visual]
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
