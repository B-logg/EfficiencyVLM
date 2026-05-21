"""
FastV visual token pruner for InternVL3.5 / Qwen3.

Reference: "An Image is Worth 1/2 Tokens After Layer 2"
           Liang Chen et al., ECCV 2024  (arXiv 2403.06764)

This module monkey-patches Qwen3DecoderLayer and Qwen3Model at runtime so
that no changes to the transformers library installation are needed.
Just upload this file to the server and the patch applies automatically.

True mid-prefill pruning (identical to paper):
  Layers 0..K  : full sequence (256 visual tokens)
  Layer K      : attention extracted → importance scored → hidden states pruned
                 KV cache layers 0..K pruned in-place
  Layers K+1..N: pruned sequence (128 visual tokens)
  Decode       : pruned KV cache → every step attends to shorter sequence

Settings: prune_layer=8, keep_ratio=0.5  →  256 → 128 visual tokens
"""
import time
from typing import Optional, Tuple

import torch

# ── Monkey-patch state ────────────────────────────────────────────────────────

_PATCH_APPLIED = False


def _fastv_prune_kv(cache, keep_seq: torch.Tensor, n_layers: int) -> None:
    """
    Prune DynamicCache layers 0..n_layers-1 to keep_seq positions.
    DynamicLayer stores accumulated K/V in self.keys and self.values.
    """
    for i in range(min(n_layers, len(cache.layers))):
        inner = cache.layers[i]
        if hasattr(inner, "keys") and isinstance(inner.keys, torch.Tensor):
            inner.keys   = inner.keys[:, :, keep_seq, :]
            inner.values = inner.values[:, :, keep_seq, :]


def _apply_fastv_patch() -> None:
    """
    Monkey-patch Qwen3DecoderLayer.forward and Qwen3Model.forward once.

    DecoderLayer gains an `output_attentions` parameter so attention weights
    can be retrieved at layer K without a separate forward pass.

    Qwen3Model gains `fastv_*` parameters that trigger mid-prefill pruning.
    When these are absent (normal inference / decode), the call falls through
    to the original forward unchanged.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        import transformers.models.qwen3.modeling_qwen3 as _m
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        # Grab module-level mask builders before patching
        _create_causal_mask = _m.create_causal_mask
        _create_sw_mask     = getattr(_m, "create_sliding_window_causal_mask", None)

        # ── Patch 1: Qwen3DecoderLayer.forward ───────────────────────────────
        def _layer_fwd(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position=None,
            position_embeddings=None,
            output_attentions: bool = False,   # ← new
            **kwargs,
        ):
            residual      = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            residual      = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            if output_attentions:
                return hidden_states, attn_weights
            return hidden_states

        _m.Qwen3DecoderLayer.forward = _layer_fwd

        # ── Patch 2: Qwen3Model.forward ───────────────────────────────────────
        _orig_fwd = _m.Qwen3Model.forward   # saved BEFORE patching → no recursion

        def _model_fwd(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            # FastV parameters — absent during normal inference / decode
            fastv_prune_layer: Optional[int] = None,
            fastv_vis_start:   Optional[int] = None,
            fastv_n_visual:    Optional[int] = None,
            fastv_n_keep:      Optional[int] = None,
            **kwargs,
        ):
            # ── Non-FastV path: delegate to original (decode, baseline, etc.) ─
            if fastv_prune_layer is None or fastv_vis_start is None:
                return _orig_fwd(
                    self,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            # ── FastV path: mid-prefill pruning ──────────────────────────────
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if cache_position is None:
                past_seen = past_key_values.get_seq_length() if past_key_values else 0
                cache_position = torch.arange(
                    past_seen, past_seen + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # Build causal mask for full sequence
            mk = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": _create_causal_mask(**mk)}
            if self.has_sliding_layers and _create_sw_mask is not None:
                causal_mask_mapping["sliding_attention"] = _create_sw_mask(**mk)

            hidden_states       = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(
                self.layers[: self.config.num_hidden_layers]
            ):
                # ── FastV pruning at layer K ─────────────────────────────────
                if layer_idx == fastv_prune_layer:
                    hidden_states, attn_weights = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        output_attentions=True,
                        **kwargs,
                    )

                    # Importance: column-mean over all heads and all query positions
                    seq_len    = hidden_states.shape[1]
                    vis_end    = fastv_vis_start + fastv_n_visual
                    to_vis     = attn_weights[0, :, :, fastv_vis_start:vis_end]  # [H,seq,nv]
                    importance = to_vis.mean(dim=(0, 1))                          # [n_visual]
                    keep_local = importance.topk(fastv_n_keep).indices.sort().values

                    keep_seq = torch.cat([
                        torch.arange(fastv_vis_start, device=hidden_states.device),
                        keep_local + fastv_vis_start,
                        torch.arange(vis_end, seq_len, device=hidden_states.device),
                    ])
                    new_seq_len = int(keep_seq.shape[0])

                    # Prune hidden states → K+1..N see shorter sequence
                    hidden_states = hidden_states[:, keep_seq, :]

                    # Prune KV entries for layers 0..K
                    if use_cache and past_key_values is not None:
                        _fastv_prune_kv(past_key_values, keep_seq, layer_idx + 1)

                    # Rebuild position_ids with contiguous indices so the decode
                    # token at position new_seq_len follows naturally (no backward
                    # jump in RoPE relative distances).
                    position_ids   = torch.arange(
                        new_seq_len, device=hidden_states.device
                    ).unsqueeze(0)
                    cache_position = torch.arange(new_seq_len, device=hidden_states.device)

                    # Rebuild causal mask for the pruned sequence
                    mk_p = {
                        "config": self.config,
                        "input_embeds": hidden_states,
                        "attention_mask": torch.ones(
                            1, new_seq_len,
                            device=hidden_states.device,
                            dtype=inputs_embeds.dtype,
                        ),
                        "cache_position": cache_position,
                        "past_key_values": None,   # K+1..N have no prior cache
                        "position_ids": position_ids,
                    }
                    causal_mask_mapping = {
                        "full_attention": _create_causal_mask(**mk_p)
                    }
                    if self.has_sliding_layers and _create_sw_mask is not None:
                        causal_mask_mapping["sliding_attention"] = _create_sw_mask(**mk_p)

                    position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    continue   # layer K already processed above

                # ── Normal layer ─────────────────────────────────────────────
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            )

        _m.Qwen3Model.forward = _model_fwd

        _PATCH_APPLIED = True
        print("[FastV] Qwen3 monkey-patch applied: mid-prefill pruning enabled.")

    except Exception as e:
        print(f"[FastV] Warning: patch failed ({e}). Sparse pipeline will error.")
        raise


# ── FastVPruner ───────────────────────────────────────────────────────────────

class FastVPruner:
    """
    Args:
        keep_ratio  : fraction of visual tokens to keep (0.5 → 256→128)
        prune_layer : LLM layer index at which pruning occurs (0-indexed)
    """

    def __init__(self, keep_ratio: float = 0.5, prune_layer: int = 8):
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio  = keep_ratio
        self.prune_layer = prune_layer
        _apply_fastv_patch()   # ensure patch is live before first use

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
        Single prefill with mid-prefill pruning → pruned KV cache → first token.

        Returns:
            first_token_id  : [1, 1]        first generated token ID
            past_key_values : DynamicCache   all-layer KV cache (pruned)
            pruned_f_mask   : [1, new_seq]   attention mask for decode
            t_ms            : wall-clock time in milliseconds
        """
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        n_keep     = max(1, int(n_visual * self.keep_ratio))
        base_model = getattr(lm_model, "model", lm_model)

        # Single forward: Qwen3Model.forward handles all pruning internally.
        # last_hidden_state is already normed and has shape [1, new_seq_len, D].
        out = base_model(
            inputs_embeds=f_embs,
            attention_mask=f_mask,
            use_cache=True,
            return_dict=True,
            fastv_prune_layer=self.prune_layer,
            fastv_vis_start=vis_start,
            fastv_n_visual=n_visual,
            fastv_n_keep=n_keep,
        )

        # First token from the pruned sequence's last-position logits
        logits      = lm_model.lm_head(out.last_hidden_state[:, -1:, :])  # [1,1,V]
        first_token = logits.argmax(dim=-1)                                # [1, 1]

        new_seq_len = out.last_hidden_state.shape[1]
        pruned_mask = torch.ones(1, new_seq_len, dtype=f_mask.dtype, device=device)

        torch.cuda.synchronize()
        t_ms = (time.perf_counter() - t0) * 1000.0

        return first_token, out.past_key_values, pruned_mask, t_ms
