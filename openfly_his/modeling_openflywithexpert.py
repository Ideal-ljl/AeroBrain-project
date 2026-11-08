"""
Qwen3.5 + Expert combined model.

Prefix path: delegates entirely to Qwen3_5Model.forward() — handles image embedding,
mm_token_type_ids → compute_3d_position_ids, and the language model in one call.

Suffix path: runs the custom action Expert model with suffix embeddings.

Joint path (training): prefix and suffix embeddings are processed through layers
together (PI05-style), sharing K/V at full-attention layers and, when enabled,
passing the VLM recurrent state into the Expert at linear-attention layers.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from openfly_his.configuration_openfly import Qwen3_5WithExpertConfig
from openfly_his.modeling_expert import (
    Qwen3_5ExpertModel,
    apply_rotary_pos_emb,
    gated_delta_net_forward_with_state,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Joint attention layer (PI05-style: prefix and suffix share K/V)
# ---------------------------------------------------------------------------

import logging
_joint_attn_logger = logging.getLogger(__name__)
_joint_attn_backend_logged = False


def _log_sdpa_backend(q, k, v, attn_mask, tag=""):
    """Log which SDPA backend is selected (flash, mem_efficient, or math)."""
    global _joint_attn_backend_logged
    if _joint_attn_backend_logged:
        return
    _joint_attn_backend_logged = True
    from torch.nn.attention import sdpa_kernel, SDPBackend
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    results = []
    for backend in backends:
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(
                    q[:1], k[:1], v[:1],
                    attn_mask=attn_mask[:1] if attn_mask is not None and attn_mask.ndim >= 1 else attn_mask,
                )
            results.append(backend.name)
        except RuntimeError:
            pass
    _joint_attn_logger.warning(
        "[JointAttn] %s | q=%s k=%s dtype=%s | available backends: %s",
        tag, list(q.shape), list(k.shape), q.dtype, results,
    )


def _has_internal_or_left_padding(attention_mask: Optional[torch.Tensor]) -> bool:
    """Return True when valid prefix tokens appear after masked positions."""
    if attention_mask is None:
        return False
    mask = attention_mask.bool()
    if mask.ndim != 2:
        return True
    seen_pad = (~mask).cumsum(dim=-1) > 0
    return bool((seen_pad & mask).any().item())


def _make_prefix_causal_padding_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if attention_mask is None or not _has_internal_or_left_padding(attention_mask):
        return None

    attention_mask = attention_mask.to(device=device)
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            "prefix_attention_mask shape mismatch: "
            f"expected {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}"
        )

    min_val = -1e4
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
    causal_block = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = mask.masked_fill(causal_block[None, None, :, :], min_val)
    key_padding = ~attention_mask.bool()
    mask = mask.masked_fill(key_padding[:, None, None, :], min_val)
    return mask


def compute_joint_full_attn_layer(
    vlm_layer, expert_layer,
    prefix_hidden, suffix_hidden,
    prefix_pos_embs, suffix_pos_embs,
    prefix_causal_mask, suffix_joint_mask,
):
    """
    One full_attention layer with joint K/V sharing between VLM (prefix) and Expert (suffix).

    - Prefix Q attends only to prefix K/V (causal, no cross-attention to suffix).
    - Suffix Q attends to concatenated [prefix K/V, suffix K/V].
    - Uses F.scaled_dot_product_attention (SDPA) for memory-efficient attention.
    """
    # ─── LayerNorm ────────────────────────────────────────────
    prefix_residual = prefix_hidden
    prefix_normed = vlm_layer.input_layernorm(prefix_hidden)

    suffix_residual = suffix_hidden
    suffix_normed = expert_layer.input_layernorm(suffix_hidden)

    # ─── Q / K / V  for VLM prefix ───────────────────────────
    vlm_attn = vlm_layer.self_attn
    p_shape = prefix_normed.shape[:-1]                          # [B, S_prefix]
    p_h_shape = (*p_shape, -1, vlm_attn.head_dim)

    p_q_raw, p_gate = torch.chunk(
        vlm_attn.q_proj(prefix_normed).view(*p_shape, -1, vlm_attn.head_dim * 2), 2, dim=-1
    )
    p_gate = p_gate.reshape(*p_shape, -1)
    p_q = vlm_attn.q_norm(p_q_raw.view(p_h_shape)).transpose(1, 2)
    p_k = vlm_attn.k_norm(vlm_attn.k_proj(prefix_normed).view(p_h_shape)).transpose(1, 2)
    p_v = vlm_attn.v_proj(prefix_normed).view(p_h_shape).transpose(1, 2)

    p_cos, p_sin = prefix_pos_embs
    p_q, p_k = apply_rotary_pos_emb(p_q, p_k, p_cos, p_sin)

    # ─── Q / K / V  for Expert suffix ────────────────────────
    expert_attn = expert_layer.self_attn
    s_shape = suffix_normed.shape[:-1]                          # [B, S_suffix]
    s_h_shape = (*s_shape, -1, expert_attn.head_dim)

    s_q_raw, s_gate = torch.chunk(
        expert_attn.q_proj(suffix_normed).view(*s_shape, -1, expert_attn.head_dim * 2), 2, dim=-1
    )
    s_gate = s_gate.reshape(*s_shape, -1)
    s_q = expert_attn.q_norm(s_q_raw.view(s_h_shape)).transpose(1, 2)
    s_k = expert_attn.k_norm(expert_attn.k_proj(suffix_normed).view(s_h_shape)).transpose(1, 2)
    s_v = expert_attn.v_proj(suffix_normed).view(s_h_shape).transpose(1, 2)

    s_cos, s_sin = suffix_pos_embs
    s_q, s_k = apply_rotary_pos_emb(s_q, s_k, s_cos, s_sin)

    # ─── Prefix self-attention ──────────────────────────────────
    # Keep the Flash-friendly causal fast path for all-valid/right-padded
    # prefixes. MAM compaction can create internal padding inside the prepended
    # memory prefix, where valid text tokens follow masked memory slots; that
    # case needs an explicit causal + key-padding mask to match native Qwen.
    p_k_exp = repeat_kv(p_k, vlm_attn.num_key_value_groups)
    p_v_exp = repeat_kv(p_v, vlm_attn.num_key_value_groups)

    if prefix_causal_mask is None:
        p_attn_out = torch.nn.functional.scaled_dot_product_attention(
            p_q, p_k_exp, p_v_exp,
            is_causal=True,
            scale=vlm_attn.scaling,
        )
    else:
        p_attn_out = torch.nn.functional.scaled_dot_product_attention(
            p_q, p_k_exp, p_v_exp,
            attn_mask=prefix_causal_mask,
            scale=vlm_attn.scaling,
        )
    p_attn_out = p_attn_out.transpose(1, 2).contiguous().reshape(*p_shape, -1)
    p_attn_out = p_attn_out * torch.sigmoid(p_gate)
    p_attn_out = vlm_attn.o_proj(p_attn_out)

    # ─── Suffix cross+self attention (SDPA) ───────────────────
    # Suffix is very short (~chunk_size tokens), so the attention matrix
    # [B, H, S_suffix, S_prefix+S_suffix] is small even with explicit mask.
    k_full = torch.cat([p_k, s_k], dim=2)
    v_full = torch.cat([p_v, s_v], dim=2)
    k_full_exp = repeat_kv(k_full, expert_attn.num_key_value_groups)
    v_full_exp = repeat_kv(v_full, expert_attn.num_key_value_groups)

    s_attn_out = torch.nn.functional.scaled_dot_product_attention(
        s_q, k_full_exp, v_full_exp,
        attn_mask=suffix_joint_mask,
        scale=expert_attn.scaling,
    )
    s_attn_out = s_attn_out.transpose(1, 2).contiguous().reshape(*s_shape, -1)
    s_attn_out = s_attn_out * torch.sigmoid(s_gate)
    s_attn_out = expert_attn.o_proj(s_attn_out)

    # ─── Residuals ────────────────────────────────────────────
    prefix_hidden = prefix_residual + p_attn_out
    suffix_hidden = suffix_residual + s_attn_out

    # ─── MLP ──────────────────────────────────────────────────
    prefix_residual = prefix_hidden
    prefix_hidden = vlm_layer.post_attention_layernorm(prefix_hidden)
    prefix_hidden = vlm_layer.mlp(prefix_hidden)
    prefix_hidden = prefix_residual + prefix_hidden

    suffix_residual = suffix_hidden
    suffix_hidden = expert_layer.post_attention_layernorm(suffix_hidden)
    suffix_hidden = expert_layer.mlp(suffix_hidden)
    suffix_hidden = suffix_residual + suffix_hidden

    return prefix_hidden, suffix_hidden


def compute_joint_linear_attn_layer(
    vlm_layer, expert_layer,
    prefix_hidden, suffix_hidden,
    prefix_linear_mask, suffix_linear_mask,
    share_linear_attention_state=False,
):
    """Run one paired Gated DeltaNet layer, optionally handing off prefix state."""
    # VLM prefix
    p_res = prefix_hidden
    prefix_hidden = vlm_layer.input_layernorm(prefix_hidden)
    prefix_recurrent_state = None
    if share_linear_attention_state:
        prefix_hidden, prefix_recurrent_state = gated_delta_net_forward_with_state(
            vlm_layer.linear_attn,
            prefix_hidden,
            attention_mask=prefix_linear_mask,
            return_final_state=True,
        )
    else:
        prefix_hidden = vlm_layer.linear_attn(
            hidden_states=prefix_hidden,
            cache_params=None,
            attention_mask=prefix_linear_mask,
        )
    prefix_hidden = p_res + prefix_hidden
    p_res = prefix_hidden
    prefix_hidden = vlm_layer.post_attention_layernorm(prefix_hidden)
    prefix_hidden = vlm_layer.mlp(prefix_hidden)
    prefix_hidden = p_res + prefix_hidden

    # Expert suffix
    s_res = suffix_hidden
    suffix_hidden = expert_layer.input_layernorm(suffix_hidden)
    if share_linear_attention_state:
        suffix_hidden, _ = gated_delta_net_forward_with_state(
            expert_layer.linear_attn,
            suffix_hidden,
            attention_mask=suffix_linear_mask,
            initial_recurrent_state=prefix_recurrent_state,
            return_final_state=False,
        )
    else:
        suffix_hidden = expert_layer.linear_attn(
            hidden_states=suffix_hidden,
            cache_params=None,
            attention_mask=suffix_linear_mask,
        )
    suffix_hidden = s_res + suffix_hidden
    s_res = suffix_hidden
    suffix_hidden = expert_layer.post_attention_layernorm(suffix_hidden)
    suffix_hidden = expert_layer.mlp(suffix_hidden)
    suffix_hidden = s_res + suffix_hidden

    return prefix_hidden, suffix_hidden


class Qwen3_5WithExpertModel(PreTrainedModel):
    config_class = Qwen3_5WithExpertConfig
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: Qwen3_5WithExpertConfig):
        super().__init__(config)
        self.config = config
        self.vlm_backbone = Qwen3_5ForConditionalGeneration(config=self.config.vlm_config)
        router_num_clusters = getattr(config, "num_gmm_clusters", 17)
        if router_num_clusters is None:
            router_num_clusters = 17
        self.expert = Qwen3_5ExpertModel(
            config.expert_config,
            config.action_dim,
            config.state_dim,
            config.vlm_config.text_config.hidden_size,
            int(router_num_clusters),
            share_linear_attention_state=getattr(config, "share_linear_attention_state", False),
        )
        self.post_init()

    # ------------------------------------------------------------------
    # Joint forward (PI05-style: prefix + suffix through layers together)
    # ------------------------------------------------------------------

    def forward_joint(
        self,
        prefix_embeds: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor],
        prefix_action_attention_mask: Optional[torch.Tensor],
        suffix_embeds: torch.Tensor,
        suffix_position_ids: torch.Tensor,
    ) -> list:
        """
        PI05-style joint forward: prefix and suffix go through transformer
        layers together.

        * Full-attention layers: suffix Q attends to concatenated
          [prefix K/V, suffix K/V].  Prefix Q attends only to prefix K/V
          (causal, same as standalone VLM).
        * Linear-attention layers: the branches keep separate projections and
          convolutions; when configured, the Expert recurrent kernel starts
          from the corresponding VLM prefix state.

        Returns: [prefix_hidden_states, suffix_hidden_states]
        """
        vlm_text = self.vlm_backbone.model.language_model   # Qwen3_5TextModel
        expert = self.expert
        num_layers = vlm_text.config.num_hidden_layers

        B = prefix_embeds.shape[0]
        S_prefix = prefix_embeds.shape[1]
        S_suffix = suffix_embeds.shape[1]
        device = prefix_embeds.device
        dtype = prefix_embeds.dtype

        # ── Prepare position IDs & rotary embeddings ─────────────
        # prefix_position_ids: [3, B, S_prefix] from compute_3d_position_ids
        #                      or None (text-only, no images)
        if prefix_position_ids is None:
            seq_ids = torch.arange(S_prefix, device=device)
            prefix_position_ids = seq_ids.view(1, 1, -1).expand(3, B, S_prefix)

        if prefix_position_ids.ndim == 2:
            prefix_mrope_ids = prefix_position_ids[None].expand(3, B, S_prefix)
        else:
            prefix_mrope_ids = prefix_position_ids  # [3, B, S]

        prefix_pos_embs = vlm_text.rotary_emb(prefix_embeds, prefix_mrope_ids)

        # suffix: [B, S_suffix] → expand to [3, B, S_suffix] for rotary
        if suffix_position_ids.ndim == 2:
            suffix_mrope_ids = suffix_position_ids[None].expand(3, B, S_suffix)
        elif suffix_position_ids.ndim == 3:
            suffix_mrope_ids = suffix_position_ids
        else:
            suffix_mrope_ids = suffix_position_ids

        suffix_pos_embs = expert.rotary_emb(suffix_embeds, suffix_mrope_ids)

        # ── Build attention masks ─────────────────────────────────
        prefix_causal_mask = _make_prefix_causal_padding_mask(
            prefix_attention_mask,
            batch_size=B,
            seq_len=S_prefix,
            device=device,
            dtype=dtype,
        )

        # Suffix joint mask [B, 1, S_suffix, S_prefix + S_suffix].
        # The prefix part uses an action-specific mask so AR grounding answer
        # tokens can train the VLM without being visible to action suffix tokens.
        # Match the two-stage suffix mask. bf16 + flash attention can produce
        # NaN gradients with finfo(dtype).min even when the forward loss is
        # finite; a large finite negative value is enough to mask attention.
        min_val = -1e4
        visible_prefix_mask = prefix_action_attention_mask
        if visible_prefix_mask is None:
            visible_prefix_mask = prefix_attention_mask
        if visible_prefix_mask is not None:
            sees_prefix = (~visible_prefix_mask.bool()).to(dtype) * min_val
            sees_prefix = sees_prefix[:, None, None, :].expand(B, 1, S_suffix, S_prefix)
        else:
            sees_prefix = torch.zeros(B, 1, S_suffix, S_prefix, device=device, dtype=dtype)
        suffix_full = torch.zeros(B, 1, S_suffix, S_suffix, device=device, dtype=dtype)
        suffix_joint_mask = torch.cat([sees_prefix, suffix_full], dim=-1)

        # ── Linear-attention masks ───────────────────────────────
        prefix_linear_mask = vlm_text._update_linear_attn_mask(prefix_attention_mask, None)
        suffix_linear_mask = expert._update_linear_attn_mask(None, None)
        share_linear_attention_state = bool(
            getattr(self.config, "share_linear_attention_state", False)
        )

        # ── Gradient-checkpointing flag ──────────────────────────
        use_gc = self.training and (
            getattr(vlm_text, "gradient_checkpointing", False)
            or getattr(expert, "gradient_checkpointing", False)
        )

        # ── Layer-by-layer forward ───────────────────────────────
        prefix_hidden = prefix_embeds
        suffix_hidden = suffix_embeds

        for layer_idx in range(num_layers):
            vlm_layer = vlm_text.layers[layer_idx]
            expert_layer = expert.layers[layer_idx]
            layer_type = vlm_text.config.layer_types[layer_idx]

            if layer_type == "full_attention":
                if use_gc:
                    prefix_hidden, suffix_hidden = torch.utils.checkpoint.checkpoint(
                        compute_joint_full_attn_layer,
                        vlm_layer, expert_layer,
                        prefix_hidden, suffix_hidden,
                        prefix_pos_embs, suffix_pos_embs,
                        prefix_causal_mask, suffix_joint_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    prefix_hidden, suffix_hidden = compute_joint_full_attn_layer(
                        vlm_layer, expert_layer,
                        prefix_hidden, suffix_hidden,
                        prefix_pos_embs, suffix_pos_embs,
                        prefix_causal_mask, suffix_joint_mask,
                    )
            else:
                # ── Linear attention: optionally hand off prefix state ──
                if use_gc:
                    prefix_hidden, suffix_hidden = torch.utils.checkpoint.checkpoint(
                        compute_joint_linear_attn_layer,
                        vlm_layer, expert_layer,
                        prefix_hidden, suffix_hidden,
                        prefix_linear_mask, suffix_linear_mask,
                        share_linear_attention_state,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    prefix_hidden, suffix_hidden = compute_joint_linear_attn_layer(
                        vlm_layer, expert_layer,
                        prefix_hidden, suffix_hidden,
                        prefix_linear_mask, suffix_linear_mask,
                        share_linear_attention_state,
                    )

        # ── Final norms ──────────────────────────────────────────
        prefix_hidden = vlm_text.norm(prefix_hidden)
        suffix_hidden = expert.norm(suffix_hidden)

        return [prefix_hidden, suffix_hidden]

    # ------------------------------------------------------------------
    # Original two-pass forward (kept for inference / backward compat)
    # ------------------------------------------------------------------

    def forward(
        self,
        # Raw VLM inputs (prefix path)
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # Suffix-path embedding (set to non-None for expert-only forward)
        suffix_embeds: Optional[torch.FloatTensor] = None,
        full_suffix_attention: bool = False,
        past_key_values_seq_length: Optional[int] = None,
    ) -> Tuple:
        """
        Returns: ([prefix_hidden_states, suffix_hidden_states], past_key_values, rope_deltas)

        Prefix mode  (suffix_embeds is None):
            Runs the full Qwen3.5 VLM pipeline (image embedding + MRoPE + language model).
        Suffix mode  (suffix_embeds is not None, input_ids is None):
            Runs only the action Expert.
        """

        if suffix_embeds is None:
            # ── Prefix: let Qwen3.5 handle everything natively ──────────────
            vlm_output = self.vlm_backbone.model(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                use_cache=use_cache,
            )
            prefix_hidden = vlm_output.last_hidden_state
            prefix_pkv = vlm_output.past_key_values
            rope_deltas = vlm_output.rope_deltas
            return [prefix_hidden, None], prefix_pkv, rope_deltas

        else:
            # ── Suffix: run Expert only ──────────────────────────────────────
            expert_out = self.expert(
                inputs_embeds=suffix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                full_suffix_attention=full_suffix_attention,
                past_key_values_seq_length=past_key_values_seq_length,
            )
            return [None, expert_out.last_hidden_state], None, None
