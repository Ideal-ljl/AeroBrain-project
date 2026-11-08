"""
Expert model for OpenFLY, adapted for Qwen3.5 architecture.

Key differences from Qwen3-VL expert:
- Standard 2D RoPE (not MRoPE with 3D T/H/W position IDs)
- q_proj outputs head_dim*2, split into query + gate; gate applied after o_proj
- RMSNorm uses (1 + weight) style (zero-init weight)
- position_ids are 2D (batch, seq_len) instead of 3D
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    apply_mask_to_padding_states,
)


def gated_delta_net_forward_with_state(
    module: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    initial_recurrent_state: Optional[torch.Tensor] = None,
    return_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run a Qwen3.5 Gated DeltaNet chunk with an explicit initial state.

    Transformers only consumes a cached linear-attention state when the query
    length is one. OpenFLY's action suffix is a state token plus an action
    chunk, so the stock cache path silently starts that suffix from an empty
    state. This function keeps the native projections, causal convolution,
    gated-delta kernel, norm, and output projection, while exposing the
    kernel's existing ``initial_state`` argument for a multi-token suffix.

    The convolution is intentionally local to each branch. VLM and Expert
    have different convolution weights, so only the associative recurrent
    state is handed off between them.
    """
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    z = module.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, module.head_v_dim)
    beta = module.in_proj_b(hidden_states).sigmoid()
    a = module.in_proj_a(hidden_states)

    if module.causal_conv1d_fn is not None:
        mixed_qkv = module.causal_conv1d_fn(
            x=mixed_qkv,
            weight=module.conv1d.weight.squeeze(1),
            bias=module.conv1d.bias,
            activation=module.activation,
            seq_idx=None,
        )
    else:
        mixed_qkv = F.silu(module.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

    # Padding steps must not evolve the recurrent state. `g` and `beta` are
    # non-zero even where the hidden state was zeroed (`in_proj_a`/`in_proj_b`
    # feed softplus/sigmoid), so with the right-padding this repo uses, the
    # returned final state is the real end-of-sequence state decayed by n_pad
    # extra steps — a per-sample-random attenuation of the handoff. g=0 makes
    # the decay factor exp(0)=1 and beta=0 drops the delta update, so a pad
    # step becomes an identity. Causality keeps real positions untouched.
    if (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape[1] == seq_len
    ):
        keep = attention_mask[:, :, None]
        g = g * keep.to(g.dtype)
        beta = beta * keep.to(beta.dtype)

    if module.num_v_heads // module.num_k_heads > 1:
        repeat = module.num_v_heads // module.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core_attn_out, final_recurrent_state = module.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_recurrent_state,
        output_final_state=return_final_state,
        use_qk_l2norm_in_kernel=True,
    )

    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return module.out_proj(core_attn_out), final_recurrent_state


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Qwen3.5 uses partial rotary: cos/sin may cover only the first rotary_dim elements
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Qwen3_5ExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    """Expert decoder layer: same as Qwen3.5 text decoder."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        linear_initial_recurrent_state: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        # ── Self-attention (or linear attention) ────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            if linear_initial_recurrent_state is None:
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
            else:
                hidden_states, _ = gated_delta_net_forward_with_state(
                    self.linear_attn,
                    hidden_states,
                    attention_mask=attention_mask,
                    initial_recurrent_state=linear_initial_recurrent_state,
                    return_final_state=False,
                )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # ── FFN ─────────────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
    
# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExpertModelOutputWithPast(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# ---------------------------------------------------------------------------
# Pre-trained model base
# ---------------------------------------------------------------------------

class Qwen3_5ExpertPreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5ExpertDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True


# ---------------------------------------------------------------------------
# ActionRouterHead
# ---------------------------------------------------------------------------

class ActionRouterHead(nn.Module):
    def __init__(self, hidden_size, n_clusters, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, 4 * hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Linear(4 * hidden_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, n_clusters)

    def forward(self, h_t):
        feat = self.mlp(h_t)
        logits = self.classifier(feat)
        logits = torch.softmax(logits, dim=-1)
        return logits


# ---------------------------------------------------------------------------
# Expert Model
# ---------------------------------------------------------------------------

class Qwen3_5ExpertModel(Qwen3_5TextModel):
    """
    Action expert decoder for OpenFLY, built on Qwen3.5 text architecture.
    Mirrors the VLM's hybrid full/linear-attention layer schedule.
    Uses standard 2D RoPE (no MRoPE).
    """

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        action_dim,
        state_dim,
        vlm_dim,
        router_num_clusters=17,
        share_linear_attention_state=False,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 删除从父类继承的 embed_tokens，expert 只接收 inputs_embeds
        del self.embed_tokens
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.share_linear_attention_state = bool(share_linear_attention_state)




        self.action_in_proj = nn.Linear(action_dim, config.hidden_size)
        self.state_proj = nn.Linear(state_dim, config.hidden_size)

        self.action_time_mlp_in = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.action_time_mlp_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.action_out_proj = nn.Linear(config.hidden_size, action_dim)

        self.router_head = ActionRouterHead(vlm_dim, router_num_clusters)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        full_suffix_attention: bool = False,
        past_key_values_seq_length: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            if past_key_values_seq_length is not None:
                past_seen_tokens = past_key_values_seq_length
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None


        if full_suffix_attention and past_key_values is not None:
            batch_size, query_length = inputs_embeds.shape[:2]
            prefix_key_value_length = (
                past_key_values_seq_length
                if past_key_values_seq_length is not None
                else past_key_values.get_seq_length()
            )
            key_value_length = prefix_key_value_length + query_length
            if attention_mask is not None:
                causal_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                if causal_mask.shape[-2:] != (query_length, key_value_length):
                    raise ValueError(
                        "full_suffix_attention mask shape mismatch: "
                        f"expected (..., {query_length}, {key_value_length}), got {tuple(causal_mask.shape)}"
                    )
            else:
                causal_mask = torch.zeros(
                    batch_size,
                    1,
                    query_length,
                    key_value_length,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
            linear_attn_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
            linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask

            is_linear_layer = self.config.layer_types[i] == "linear_attention"
            linear_initial_recurrent_state = None
            if (
                full_suffix_attention
                and is_linear_layer
                and self.share_linear_attention_state
                and past_key_values is not None
                and past_key_values.has_previous_state(i)
            ):
                linear_initial_recurrent_state = past_key_values.layers[i].recurrent_states

            # Never mutate the cached VLM linear state with a noisy action
            # suffix. Each denoising step must start from the same prefix state.
            layer_past_key_values = None if full_suffix_attention and is_linear_layer else past_key_values
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=layer_past_key_values,
                linear_initial_recurrent_state=linear_initial_recurrent_state,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
