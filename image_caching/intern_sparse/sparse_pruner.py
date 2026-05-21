"""
True single-pass FastV: pruning happens MID-PREFILL at layer K.

Reference: "An Image is Worth 1/2 Tokens After Layer 2"
           Liang Chen et al., ECCV 2024  (arXiv 2403.06764)

Exact mechanism (matches paper):
  Phase 1 — Layers 0..K, full sequence (256 visual tokens):
    Run each layer individually with use_cache=True.
    At layer K, capture attention weights.

  Prune:
    Compute importance = column-mean of attention at K over all heads
    and all query positions.
    Select top keep_ratio visual tokens.
    • Slice hidden_states to the shorter sequence.
    • Remove pruned positions from KV cache entries of layers 0..K.

  Phase 2 — Layers K+1..N, pruned sequence (128 visual tokens):
    Continue from the pruned hidden_states.
    KV cache for K+1..N is built from the shorter sequence only.
    position_ids carry the ORIGINAL positions of kept tokens
    so RoPE is applied at the correct coordinates.

  First token:
    Apply base_model.norm + lm_model.lm_head to the last hidden
    state of the pruned sequence → argmax → first generated token.

  Decode:
    generate() receives (input_ids=first_token, past_key_values=cache,
    attention_mask=pruned_mask).  Every decode step attends to a
    shorter KV cache → faster TPOT.

Settings: prune_layer=8, keep_ratio=0.5  →  256 → 128 visual tokens
"""
import inspect
import time
from typing import Optional, Tuple

import torch

try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _call_layer(layer, hidden_states: torch.Tensor, **kwargs) -> tuple:
    """
    Call a decoder layer, silently dropping kwargs the layer doesn't accept.
    Handles API differences across transformers versions (position_embeddings,
    cache_position, etc.) without requiring version checks.
    """
    sig = inspect.signature(layer.forward)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return layer(hidden_states, **valid)


def _build_causal_mask(
    base_model,
    attention_mask_2d: torch.Tensor,
    hidden_states: torch.Tensor,
    cache_position: torch.Tensor,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Build a 4D additive causal mask [1, 1, seq, seq].
    Uses the model's own _update_causal_mask first (version-safe),
    falls back to a manual lower-triangular mask.
    """
    if hasattr(base_model, "_update_causal_mask"):
        try:
            mask = base_model._update_causal_mask(
                attention_mask=attention_mask_2d,
                input_tensor=hidden_states,
                cache_position=cache_position,
                past_key_values=None,
                output_attentions=False,
            )
            if mask is not None:
                return mask
        except Exception:
            pass

    seq = hidden_states.shape[1]
    dtype = hidden_states.dtype
    mask = torch.zeros(1, 1, seq, seq, dtype=dtype, device=device)
    return mask.masked_fill(
        torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1),
        torch.finfo(dtype).min,
    )


def _get_rotary_embeds(
    base_model,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute rotary position embeddings if the model exposes rotary_emb.
    Tries common call signatures across transformers versions.
    Returns None if the model handles RoPE internally (no action needed).
    """
    if not hasattr(base_model, "rotary_emb"):
        return None
    for call in [
        lambda: base_model.rotary_emb(hidden_states, position_ids),
        lambda: base_model.rotary_emb(position_ids.shape[-1], device=hidden_states.device),
    ]:
        try:
            result = call()
            if result is not None:
                return result
        except Exception:
            continue
    return None


# ── main class ────────────────────────────────────────────────────────────────

class FastVPruner:
    """
    Args:
        keep_ratio  : fraction of visual tokens to keep (0.5 → 256→128)
        prune_layer : LLM layer index at which to prune (0-indexed)
    """

    def __init__(self, keep_ratio: float = 0.5, prune_layer: int = 8):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio  = keep_ratio
        self.prune_layer = prune_layer

    @torch.no_grad()
    def prefill_and_prune(
        self,
        lm_model,               # model.language_model  (ForCausalLM)
        f_embs: torch.Tensor,   # [1, seq_len, D_llm]
        f_mask: torch.Tensor,   # [1, seq_len]
        vis_start: int,
        n_visual: int,
        device: str,
    ) -> Tuple[torch.Tensor, object, torch.Tensor, float]:
        """
        True single-pass FastV prefill.

        Returns:
            first_token_id  : [1, 1]       token ID of first generated token
            cache           : DynamicCache  KV cache (pruned, all N layers)
            pruned_f_mask   : [1, new_seq]  2-D attention mask for decode
            t_ms            : wall-clock time in milliseconds
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        seq_len  = f_embs.shape[1]
        vis_end  = vis_start + n_visual
        n_keep   = max(1, int(n_visual * self.keep_ratio))

        base_model = getattr(lm_model, "model", lm_model)
        n_layers   = len(base_model.layers)

        # ── Phase 1 setup ─────────────────────────────────────────────────────
        hidden_states  = f_embs                                             # [1, seq, D]
        cache          = DynamicCache() if _HAS_DYNAMIC_CACHE else None
        position_ids   = torch.arange(seq_len, device=device).unsqueeze(0) # [1, seq]
        cache_pos      = torch.arange(seq_len, device=device)              # [seq]
        causal_mask    = _build_causal_mask(base_model, f_mask, hidden_states, cache_pos, device)
        pos_embeds     = _get_rotary_embeds(base_model, hidden_states, position_ids)

        attn_at_K: Optional[torch.Tensor] = None

        # ── Phase 1: layers 0..K, full sequence ───────────────────────────────
        for idx in range(self.prune_layer + 1):
            out = _call_layer(
                base_model.layers[idx],
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=(idx == self.prune_layer),
                use_cache=True,
                cache_position=cache_pos,
                position_embeddings=pos_embeds,
            )
            hidden_states = out[0]
            if idx == self.prune_layer:
                # (hidden, attn_weights, ...) when output_attentions=True
                attn_at_K = out[1] if len(out) > 1 else None

        # ── Importance scoring (FastV paper criterion) ─────────────────────────
        if attn_at_K is not None:
            to_vis     = attn_at_K[0, :, :, vis_start:vis_end]  # [H, seq, n_vis]
            importance = to_vis.mean(dim=(0, 1))                 # [n_vis]
        else:
            # Fallback: prefer earlier tokens (conservative)
            importance = torch.arange(n_visual, dtype=torch.float, device=device).flip(0)

        keep_local = importance.topk(n_keep).indices.sort().values  # within visual block
        keep_seq   = torch.cat([
            torch.arange(vis_start, device=device),
            keep_local + vis_start,
            torch.arange(vis_end, seq_len, device=device),
        ])
        new_seq_len = int(keep_seq.shape[0])

        # ── Prune hidden states ────────────────────────────────────────────────
        hidden_states = hidden_states[:, keep_seq, :]   # [1, new_seq, D]

        # ── Prune KV cache for layers 0..K ────────────────────────────────────
        if _HAS_DYNAMIC_CACHE and isinstance(cache, DynamicCache):
            for i in range(self.prune_layer + 1):
                if i < len(cache.key_cache):
                    cache.key_cache[i]   = cache.key_cache[i][:, :, keep_seq, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, keep_seq, :]
            # Update "seen tokens" counter so decode positions are correct
            for attr in ("_seen_tokens", "seen_tokens"):
                if hasattr(cache, attr):
                    setattr(cache, attr, new_seq_len)
                    break

        # ── Phase 2 setup (pruned sequence) ───────────────────────────────────
        # position_ids = original positions of kept tokens → RoPE is correct
        position_ids_p = keep_seq.unsqueeze(0)                              # [1, new_seq]
        cache_pos_p    = torch.arange(new_seq_len, device=device)           # [new_seq]
        mask_2d_p      = torch.ones(1, new_seq_len, device=device, dtype=f_mask.dtype)
        causal_mask_p  = _build_causal_mask(base_model, mask_2d_p, hidden_states, cache_pos_p, device)
        pos_embeds_p   = _get_rotary_embeds(base_model, hidden_states, position_ids_p)

        # ── Phase 2: layers K+1..N, pruned sequence ───────────────────────────
        for idx in range(self.prune_layer + 1, n_layers):
            out = _call_layer(
                base_model.layers[idx],
                hidden_states,
                attention_mask=causal_mask_p,
                position_ids=position_ids_p,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_pos_p,
                position_embeddings=pos_embeds_p,
            )
            hidden_states = out[0]

        # ── First generated token ──────────────────────────────────────────────
        # base_model.norm is the final RMSNorm; lm_head projects to vocabulary.
        hidden_states = base_model.norm(hidden_states)
        logits        = lm_model.lm_head(hidden_states[:, -1:, :])  # [1, 1, V]
        first_token   = logits.argmax(dim=-1)                        # [1, 1]

        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        return first_token, cache, mask_2d_p, t_ms
