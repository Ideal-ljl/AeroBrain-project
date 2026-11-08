"""Final v5 MSP + MAM model; checkpoint parameter names are preserved."""
from typing import Optional, Union
from contextlib import contextmanager, nullcontext
import copy
import os
import math
import warnings
import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from dataclasses import dataclass
from torch import Tensor
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from openfly_his.configuration_openfly import OpenFLYModelConfig
from openfly_his.contants import IGNORE_INDEX
from transformers.cache_utils import Cache
from pathlib import Path
from openfly_his.modeling_openflywithexpert import Qwen3_5WithExpertModel
from airbrain.memory.mam import MAMMemoryEncoder

def create_sinusoidal_pos_embedding(time: torch.tensor, dimension: int, min_period: float=0.004, max_period: float=4.0, device='cpu') -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')
    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')
    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

@dataclass
class OpenFLYOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    fm_loss: torch.FloatTensor | None = None
    ar_loss: torch.FloatTensor | None = None
    router_loss: torch.FloatTensor | None = None
    prior_nll: torch.FloatTensor | None = None
    prior_mean_mse: torch.FloatTensor | None = None
    prior_std_mean: torch.FloatTensor | None = None
    fm_loss_xyz: torch.FloatTensor | None = None
    fm_loss_stop: torch.FloatTensor | None = None
    prior_std_floor_frac: torch.FloatTensor | None = None
    prior_z_absmax: torch.FloatTensor | None = None
    actions: torch.FloatTensor | None = None
    noise: torch.FloatTensor | None = None
    suffix_hidden_states: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    meta_states: torch.FloatTensor | None = None

class OpenFLYModel(PreTrainedModel, GenerationMixin):
    config_class = OpenFLYModelConfig
    _keep_in_fp32_modules_strict = ['motion_prior']
    _supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: OpenFLYModelConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.qwen3_5_with_expert_config.vlm_config.text_config.hidden_size
        self.qwen3_5_with_expert = Qwen3_5WithExpertModel(config=config.qwen3_5_with_expert_config)
        self.expert_hidden_size = config.qwen3_5_with_expert_config.expert_config.hidden_size
        if getattr(config, 'training_objective', 'legacy') in ('expert_adapt', 'flow_joint', 'full_joint'):
            self._keep_in_fp32_modules_strict = ['motion_prior', 'qwen3_5_with_expert.expert']
        self.training_objective = 'expert_adapt'
        self.prior_loss_weight = 0.1
        self.fm_stop_weight = 1.0
        from .template_motion_prior import TemplateMotionPrior
        self.motion_prior = TemplateMotionPrior(self.hidden_size, config.motion_prior_config)
        self.mam_memory_encoder = self._make_mam_memory_encoder() if self._use_mam_memory() else None
        self.committed_action_embed = nn.Parameter(torch.zeros(self.expert_hidden_size)) if bool(getattr(config, 'use_action_prefix_conditioning', False)) else None
        self._mam_event_content_cache = {}
        self._mam_anchor_feature_cache = {}
        self._mam_prefix_cache = {}
        self._runtime_vlm_prefix_cache = {}
        self.gradient_checkpointing_enabled = True
        self.post_init()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Linear) and hasattr(module, '_source_init_bias'):
            with torch.no_grad():
                if not getattr(module.weight, '_is_hf_initialized', False):
                    nn.init.zeros_(module.weight)
                if not getattr(module.bias, '_is_hf_initialized', False):
                    nn.init.constant_(module.bias, module._source_init_bias)
        from .template_motion_prior import TemplateMotionPrior
        if isinstance(module, TemplateMotionPrior):
            module.reset_basis()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self.config, 'freeze_mam', False) and self.mam_memory_encoder is not None:
            self.mam_memory_encoder.eval()
        if getattr(self.config, 'freeze_vision_tower', False):
            self.qwen3_5_with_expert.vlm_backbone.model.visual.eval()
        if getattr(self, 'training_objective', 'legacy') not in ('legacy', 'full_joint'):
            self.qwen3_5_with_expert.vlm_backbone.eval()
            if self.mam_memory_encoder is not None:
                self.mam_memory_encoder.eval()
            if self.training_objective == 'prior_warmup':
                self.qwen3_5_with_expert.expert.eval()
            if self.training_objective == 'expert_adapt':
                self.motion_prior.eval()
        return self

    def _prefix_first_enabled(self):
        return True

    def _prefix_forward_no_grad(self):
        """full_joint reuses the prefix-first forward but backprops into the prefix.

        A conditional source has to read prefix hidden states before the suffix
        can be built, so forward_joint's layer-synchronized path cannot be used;
        running the prefix first and keeping its graph is what makes joint
        training possible at all here.
        """
        return self.training_objective != 'full_joint'

    def _history_hidden(self, hidden, input_ids, attention_mask=None):
        positions = input_ids.eq(self.config.history_token_id)
        if attention_mask is not None:
            positions = positions & attention_mask.bool()
        if not (positions.sum(-1) == 1).all():
            raise ValueError('Each action sample must have exactly one visible history token')
        indices = positions.long().argmax(-1)
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]

    def _use_mam_memory(self) -> bool:
        return bool(getattr(self.config, 'use_mam_memory', False))

    def _warn_if_padded_shared_linear_prefix(self, prefix_attention_mask) -> None:
        """Warn once if a padded prefix feeds the shared linear-attention state.

        The prefix runs through stock ``Qwen3_5GatedDeltaNet.forward``, which
        caches the state after the WHOLE padded sequence. With right padding
        that state is the real end-of-sequence state decayed by n_pad extra
        steps, so the handoff to the Expert is silently attenuated. Batch-1
        inference has no padding and is unaffected; batched inference is not.
        """
        if not getattr(self.qwen3_5_with_expert.config, 'share_linear_attention_state', False):
            return
        if prefix_attention_mask is None or getattr(self, '_warned_padded_shared_prefix', False):
            return
        if bool((prefix_attention_mask == 0).any()):
            self._warned_padded_shared_prefix = True
            warnings.warn('share_linear_attention_state=True with a padded inference prefix: the cached VLM recurrent state is decayed by the padding steps, so the state handed to the action Expert is attenuated per sample. Run inference unbatched (or with equal-length prefixes) until the prefix forward is routed through gated_delta_net_forward_with_state.', RuntimeWarning, stacklevel=2)

    def _action_prefix_enabled(self) -> bool:
        return bool(getattr(self.config, 'use_action_prefix_conditioning', False)) and getattr(self, 'committed_action_embed', None) is not None

    def _ensure_committed_action_embed(self) -> None:
        """Lazily create the committed-prefix marker (e.g. when the feature is
        switched on while resuming a checkpoint that predates it)."""
        if getattr(self, 'committed_action_embed', None) is not None:
            return
        ref = next(self.parameters(), None)
        device = ref.device if ref is not None else None
        dtype = ref.dtype if ref is not None else None
        self.committed_action_embed = nn.Parameter(torch.zeros(self.expert_hidden_size, device=device, dtype=dtype))

    def _sample_action_prefix_mask(self, batch_size: int, chunk_len: int, actions_finite: torch.Tensor, device: torch.device) -> Optional[torch.Tensor]:
        """Draw a per-sample committed-prefix length ``d`` and return a
        ``[B, chunk_len]`` bool mask of frozen positions (or ``None`` when the
        feature is off / the chunk is too short to leave a supervised tail).

        The returned mask may be all-False (every sample drew ``d=0``); we still
        return it so the ``committed_action_embed`` parameter stays on the
        autograd graph, avoiding DDP unused-parameter errors.
        """
        if not self._action_prefix_enabled():
            return None
        d_max = min(int(getattr(self.config, 'action_prefix_max_len', 0)), chunk_len - 1)
        if d_max < 1:
            return None
        p_zero = float(getattr(self.config, 'action_prefix_p_zero', 0.5))
        rand_d = torch.randint(1, d_max + 1, (batch_size,), device=device)
        use_zero = torch.rand(batch_size, device=device) < p_zero
        prefix_len = torch.where(use_zero, torch.zeros_like(rand_d), rand_d)
        positions = torch.arange(chunk_len, device=device).view(1, -1)
        frozen = positions < prefix_len.view(-1, 1)
        frozen = frozen & actions_finite.to(device=device, dtype=torch.bool)
        return frozen

    def _apply_action_prefix(self, x_t: torch.Tensor, actions_for_model: torch.Tensor, actions_finite: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Freeze a random committed prefix of the chunk to the clean action
        (RTC train-time conditioning). Returns ``(x_t, frozen_mask,
        supervised_mask)`` where ``supervised_mask`` is the set of positions
        that still contribute to the flow-matching loss.
        """
        frozen_mask = self._sample_action_prefix_mask(actions_for_model.shape[0], actions_for_model.shape[1], actions_finite, device)
        if frozen_mask is None:
            if not self._action_prefix_enabled():
                return (x_t, None, actions_finite)
            frozen_mask = torch.zeros(actions_for_model.shape[0], actions_for_model.shape[1], dtype=torch.bool, device=device)
        frozen_vals = actions_for_model
        noise_std = float(getattr(self.config, 'action_prefix_noise_std', 0.0))
        if noise_std > 0.0:
            frozen_vals = actions_for_model + noise_std * self.sample_noise(actions_for_model.shape, device)
        x_t = torch.where(frozen_mask.unsqueeze(-1), frozen_vals.to(x_t.dtype), x_t)
        supervised_mask = actions_finite & ~frozen_mask
        return (x_t, frozen_mask, supervised_mask)

    @staticmethod
    def _to_if_needed(tensor: Optional[torch.Tensor], *, device=None, dtype: Optional[torch.dtype]=None) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        kwargs = {}
        if device is not None:
            target_device = torch.device(device)
            if tensor.device != target_device:
                kwargs['device'] = target_device
        if dtype is not None and tensor.dtype != dtype:
            kwargs['dtype'] = dtype
        return tensor.to(**kwargs) if kwargs else tensor

    def _make_mam_memory_encoder(self) -> MAMMemoryEncoder:
        return MAMMemoryEncoder(action_dim=int(self.config.action_dim), d_model=self.hidden_size, patch_dim=self.hidden_size, chunk_size=int(True), history_steps=int(getattr(self.config, 'mam_history_steps', 48)), m_max=int(getattr(self.config, 'mam_m_max', 16)), tokens_per_event=int(getattr(self.config, 'mam_tokens_per_event', 2)), nms_gap=int(getattr(self.config, 'mam_nms_gap', -1)), tau_percentile=float(getattr(self.config, 'mam_tau_percentile', 80.0)), lambda_visual=float(getattr(self.config, 'mam_lambda_visual', 1.0)), num_heads=int(getattr(self.config, 'mam_num_heads', 4)), num_layers=int(getattr(self.config, 'mam_num_layers', 1)), dropout=float(getattr(self.config, 'mam_dropout', 0.0)), detach_features=bool(getattr(self.config, 'mam_detach_features', True)), chunk_boundary_mode=str(getattr(self.config, 'mam_chunk_boundary_mode', 'first_only')))

    def _get_runtime_vlm_prefix_cache(self) -> dict:
        cache = getattr(self, '_runtime_vlm_prefix_cache', None)
        if cache is None:
            cache = {}
            self._runtime_vlm_prefix_cache = cache
        return cache

    @staticmethod
    def _tensor_identity_key(tensor: Optional[torch.Tensor]) -> Optional[tuple]:
        if tensor is None:
            return None
        base = tensor
        while getattr(base, '_base', None) is not None:
            base = base._base
        return (tuple((int(x) for x in tensor.shape)), str(tensor.dtype), str(tensor.device), int(base.data_ptr()), int(getattr(base, '_version', 0)), int(tensor.storage_offset()), tuple((int(x) for x in tensor.stride())))

    def _vlm_prefix_cache_key(self, *, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.Tensor], mam_prefix_cache_key: Optional[tuple], pixel_values: Optional[torch.Tensor], pixel_values_videos: Optional[torch.Tensor], image_grid_thw: Optional[torch.Tensor], video_grid_thw: Optional[torch.Tensor], mm_token_type_ids: Optional[torch.Tensor]) -> tuple:
        return ('vlm_prefix_v1', mam_prefix_cache_key, self._tensor_identity_key(input_ids), self._tensor_identity_key(attention_mask), self._tensor_identity_key(position_ids), self._tensor_identity_key(pixel_values), self._tensor_identity_key(pixel_values_videos), self._tensor_identity_key(image_grid_thw), self._tensor_identity_key(video_grid_thw), self._tensor_identity_key(mm_token_type_ids))

    @staticmethod
    def _clone_cache_for_reuse(cache):
        return copy.deepcopy(cache) if cache is not None else None

    def _store_runtime_vlm_prefix_cache(self, key: tuple, value: tuple) -> None:
        cache = self._get_runtime_vlm_prefix_cache()
        max_entries = int(getattr(self.config, 'runtime_vlm_prefix_cache_size', 0))
        if max_entries <= 0:
            return
        while len(cache) >= max_entries:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def clear_mam_event_cache(self) -> None:
        """Clear inference-only MAM caches for a new episode/sample."""
        for name in ('_mam_event_content_cache', '_mam_anchor_feature_cache', '_mam_prefix_cache', '_runtime_vlm_prefix_cache', '_runtime_mam_processor_cache', '_runtime_mam_action_norm_cache'):
            cache = getattr(self, name, None)
            if cache is not None:
                cache.clear()

    def _get_mam_event_content_cache(self) -> dict:
        cache = getattr(self, '_mam_event_content_cache', None)
        if cache is None:
            cache = {}
            self._mam_event_content_cache = cache
        return cache

    def _get_mam_anchor_feature_cache(self) -> dict:
        cache = getattr(self, '_mam_anchor_feature_cache', None)
        if cache is None:
            cache = {}
            self._mam_anchor_feature_cache = cache
        return cache

    def _get_mam_prefix_cache(self) -> dict:
        cache = getattr(self, '_mam_prefix_cache', None)
        if cache is None:
            cache = {}
            self._mam_prefix_cache = cache
        return cache

    @staticmethod
    def _mam_event_key_from_row(row) -> Optional[tuple[int, ...]]:
        values = [int(v) for v in row]
        if not values or any((v < 0 for v in values)):
            return None
        return tuple(values)

    @staticmethod
    def _mam_anchor_key_from_value(value) -> Optional[int]:
        v = int(value)
        return None if v < 0 else v

    def _mam_prefix_key_from_inputs(self, *, event_cache_keys: torch.Tensor, event_mask: torch.Tensor, prompt_ids: list[int], m_max: int, tokens_per_event: int) -> Optional[tuple]:
        if event_cache_keys.ndim != 3 or int(event_cache_keys.shape[1]) != m_max:
            return None
        batch_size = int(event_cache_keys.shape[0])
        if event_mask.ndim != 2 or tuple(event_mask.shape) != (batch_size, m_max):
            return None
        mask_rows = [[bool(x) for x in row] for row in event_mask.detach().cpu().tolist()]
        key_rows = event_cache_keys.detach().cpu().tolist()
        valid_event_keys = []
        for b, mask_row in enumerate(mask_rows):
            row_keys = []
            for k, is_valid in enumerate(mask_row):
                if not is_valid:
                    continue
                key = self._mam_event_key_from_row(key_rows[b][k])
                if key is None:
                    return None
                row_keys.append(key)
            valid_event_keys.append(tuple(row_keys))
        return ('mam_prefix_v2', batch_size, int(m_max), int(tokens_per_event), tuple((int(x) for x in prompt_ids)), tuple((tuple((int(x) for x in row)) for row in mask_rows)), tuple(valid_event_keys))

    def _finish_mam_memory_from_content(self, *, events, content_flat: torch.Tensor, ages_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        encoder = self.mam_memory_encoder
        if encoder is None:
            raise RuntimeError('mam_memory_encoder is not initialized')
        B, m_max = events.event_mask.shape
        tokens_per_event = int(encoder.tokens_per_event)
        d_model = int(encoder.d_model)
        age_tokens = encoder.action_encoder.age_tokens(ages_flat, dtype=content_flat.dtype)
        out = content_flat + age_tokens
        memory_tokens = out.reshape(B, m_max, tokens_per_event, d_model)
        memory_tokens = memory_tokens.reshape(B, m_max * tokens_per_event, d_model)
        token_mask = events.event_mask.unsqueeze(-1).expand(-1, -1, tokens_per_event)
        token_mask = token_mask.reshape(B, m_max * tokens_per_event)
        memory_tokens = memory_tokens * token_mask.unsqueeze(-1).to(memory_tokens.dtype)
        return {'tokens': memory_tokens, 'content_tokens': content_flat.reshape(B, m_max * tokens_per_event, d_model), 'mask': token_mask, 'num_events': events.num_events, 'event_indices': events.event_indices}

    def _encode_mam_memory_cached_only(self, *, history_actions: torch.Tensor, events, seg_start: torch.Tensor, seg_end: torch.Tensor, mam_event_cache_keys: torch.Tensor) -> Optional[dict[str, torch.Tensor]]:
        """Build memory tokens when every real event content token is cached."""
        encoder = self.mam_memory_encoder
        if encoder is None:
            raise RuntimeError('mam_memory_encoder is not initialized')
        B, T, _ = history_actions.shape
        m_max = int(encoder.m_max)
        tokens_per_event = int(encoder.tokens_per_event)
        d_model = int(encoder.d_model)
        if mam_event_cache_keys.ndim != 3 or mam_event_cache_keys.shape[:2] != (B, m_max):
            return None
        _, _, ages = encoder.build_event_segments(history_actions=history_actions, event_mask=events.event_mask, seg_start=seg_start, seg_end=seg_end)
        n_slots = B * m_max
        ages_flat = ages.reshape(n_slots)
        cache = self._get_mam_event_content_cache()
        key_rows = mam_event_cache_keys.detach().cpu().tolist()
        valid_indices = events.event_mask.reshape(n_slots).nonzero(as_tuple=False).flatten().cpu().tolist()
        cached_rows: list[tuple[int, torch.Tensor]] = []
        first_cached = None
        expected_shape = (tokens_per_event, d_model)
        for flat_idx in valid_indices:
            b = flat_idx // m_max
            k = flat_idx % m_max
            key = self._mam_event_key_from_row(key_rows[b][k])
            cached = cache.get(key) if key is not None else None
            if cached is None or tuple(cached.shape) != expected_shape:
                return None
            if first_cached is None:
                first_cached = cached
            cached_rows.append((flat_idx, cached))
        dtype = first_cached.dtype if first_cached is not None else next(encoder.parameters()).dtype
        content_flat = history_actions.new_zeros(n_slots, tokens_per_event, d_model, dtype=dtype)
        for flat_idx, cached in cached_rows:
            content_flat[flat_idx] = cached.to(device=content_flat.device, dtype=content_flat.dtype)
        return self._finish_mam_memory_from_content(events=events, content_flat=content_flat, ages_flat=ages_flat)

    def _select_mam_anchor_inputs(self, *, anchor_pv: torch.Tensor, anchor_grid: torch.Tensor, anchor_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a subset of processor outputs by image index."""
        if not anchor_indices:
            raise ValueError('anchor_indices must be non-empty')
        idx_t = torch.tensor(anchor_indices, dtype=torch.long, device=anchor_grid.device)
        selected_grid = anchor_grid.index_select(0, idx_t)
        total_anchors = int(anchor_grid.shape[0])
        if anchor_pv.shape[0] == total_anchors:
            return (anchor_pv.index_select(0, idx_t.to(anchor_pv.device)), selected_grid)
        counts = anchor_grid.detach().to('cpu', dtype=torch.long).prod(dim=-1).tolist()
        if len(counts) == total_anchors and sum(counts) == int(anchor_pv.shape[0]):
            chunks = list(torch.split(anchor_pv, counts, dim=0))
            selected = torch.cat([chunks[i] for i in anchor_indices], dim=0)
            return (selected, selected_grid)
        raise ValueError(f'Cannot select MAM anchor pixel_values by image index: pixel_values shape={tuple(anchor_pv.shape)}, grid shape={tuple(anchor_grid.shape)}.')

    def _encode_mam_memory_with_cache(self, *, history_actions: torch.Tensor, events, seg_start: torch.Tensor, seg_end: torch.Tensor, endpoint_features: torch.Tensor, endpoint_patch_mask: torch.Tensor, mam_event_cache_keys: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode MAM tokens while reusing age-independent event content.

        This path is intentionally inference-only. It accepts batched cache
        keys and only encodes event slots whose age-independent content is not
        already cached.
        """
        encoder = self.mam_memory_encoder
        if encoder is None:
            raise RuntimeError('MAM event cache requested but mam_memory_encoder is not initialized')
        B, T, _ = history_actions.shape
        m_max = int(encoder.m_max)
        tokens_per_event = int(encoder.tokens_per_event)
        if mam_event_cache_keys.ndim != 3 or mam_event_cache_keys.shape[:2] != (B, m_max):
            return encoder.encode(history_actions=history_actions, history_mask=torch.ones_like(history_actions[..., 0], dtype=torch.bool), events=events, seg_start=seg_start, seg_end=seg_end, endpoint_features=endpoint_features, endpoint_patch_mask=endpoint_patch_mask)
        event_mask = events.event_mask
        seg_action_pad, seg_lengths, ages = encoder.build_event_segments(history_actions=history_actions, event_mask=event_mask, seg_start=seg_start, seg_end=seg_end)
        n_slots = B * m_max
        seg_actions_flat = seg_action_pad.reshape(n_slots, T, int(encoder.action_dim))
        seg_lengths_flat = seg_lengths.reshape(n_slots)
        ages_flat = ages.reshape(n_slots)
        endpoint_features_flat = endpoint_features.reshape(n_slots, 2, endpoint_features.shape[3], endpoint_features.shape[4])
        endpoint_patch_mask_flat = endpoint_patch_mask.reshape(n_slots, 2, endpoint_features.shape[3])
        content_flat = endpoint_features.new_zeros(n_slots, tokens_per_event, int(encoder.d_model))
        cache = self._get_mam_event_content_cache()
        key_rows = mam_event_cache_keys.detach().cpu().tolist()
        miss_indices: list[int] = []
        miss_keys: list[Optional[tuple[int, ...]]] = []
        alias_indices: list[tuple[int, int]] = []
        pending_miss_by_key: dict[tuple[int, ...], int] = {}
        valid_indices = event_mask.reshape(n_slots).nonzero(as_tuple=False).flatten().cpu().tolist()
        for flat_idx in valid_indices:
            b = flat_idx // m_max
            k = flat_idx % m_max
            key = self._mam_event_key_from_row(key_rows[b][k])
            cached = cache.get(key) if key is not None else None
            expected_shape = (tokens_per_event, int(encoder.d_model))
            if cached is not None and tuple(cached.shape) == expected_shape:
                content_flat[flat_idx] = cached.to(device=content_flat.device, dtype=content_flat.dtype)
            elif key is not None and key in pending_miss_by_key:
                alias_indices.append((flat_idx, pending_miss_by_key[key]))
            else:
                miss_indices.append(flat_idx)
                miss_keys.append(key)
                if key is not None:
                    pending_miss_by_key[key] = flat_idx
        if miss_indices:
            miss_idx_t = torch.tensor(miss_indices, dtype=torch.long, device=history_actions.device)
            miss_content = encoder.encode_event_content(segment_actions=seg_actions_flat.index_select(0, miss_idx_t), segment_lengths=seg_lengths_flat.index_select(0, miss_idx_t), endpoint_features=endpoint_features_flat.index_select(0, miss_idx_t), endpoint_patch_mask=endpoint_patch_mask_flat.index_select(0, miss_idx_t)).to(dtype=content_flat.dtype)
            content_flat.index_copy_(0, miss_idx_t, miss_content)
            for row_idx, key in enumerate(miss_keys):
                if key is not None:
                    cache[key] = miss_content[row_idx].detach().to('cpu')
        for flat_idx, source_idx in alias_indices:
            content_flat[flat_idx] = content_flat[source_idx]
        return self._finish_mam_memory_from_content(events=events, content_flat=content_flat, ages_flat=ages_flat)

    def _make_mam_memory_prefix(self, *, batch_size: int, device, mam_history_actions: Optional[torch.Tensor]=None, mam_history_mask: Optional[torch.Tensor]=None, mam_anchor_pixel_values: Optional[torch.Tensor]=None, mam_anchor_grid_thw: Optional[torch.LongTensor]=None, mam_anchor_n: Optional[torch.Tensor]=None, mam_event_indices: Optional[torch.Tensor]=None, mam_event_mask: Optional[torch.Tensor]=None, mam_seg_start: Optional[torch.Tensor]=None, mam_seg_end: Optional[torch.Tensor]=None, mam_event_anchor_start: Optional[torch.Tensor]=None, mam_event_anchor_end: Optional[torch.Tensor]=None, mam_event_cache_keys: Optional[torch.Tensor]=None, mam_anchor_cache_keys: Optional[torch.Tensor]=None) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build MAM prefix tokens. Dataset side already did saliency + event
        detection + anchor frame selection, so this only needs to:
          1. Vision tower forward on the anchor pixel_values (≤ 2·M_max frames/sample)
          2. Build endpoint_features tensor by indexing per-sample anchor features
          3. Cross-attention → M_max × tokens_per_event tokens
        Returns `(None, None)` when MAM is disabled or required inputs are absent.
        """
        self._last_mam_prefix_cache_key = None
        if not self._use_mam_memory() or self.mam_memory_encoder is None:
            return (None, None)
        if mam_history_actions is None or mam_history_mask is None:
            return (None, None)
        if mam_anchor_pixel_values is None or mam_anchor_grid_thw is None:
            return (None, None)
        if mam_event_mask is None or mam_event_anchor_start is None or mam_event_anchor_end is None:
            return (None, None)
        m_max = int(self.mam_memory_encoder.m_max)
        tokens_per_event = int(self.mam_memory_encoder.tokens_per_event)
        encoder_param = next(self.mam_memory_encoder.parameters(), None)
        encoder_dtype = encoder_param.dtype if encoder_param is not None and encoder_param.is_floating_point() else None
        history_actions = self._to_if_needed(mam_history_actions, device=device, dtype=encoder_dtype)
        history_mask = self._to_if_needed(mam_history_mask, device=device, dtype=torch.bool)
        anchor_pv = self._to_if_needed(mam_anchor_pixel_values, device=device)
        anchor_grid = self._to_if_needed(mam_anchor_grid_thw, device=device)
        anchor_n = self._to_if_needed(mam_anchor_n, device=device, dtype=torch.long)
        event_mask = self._to_if_needed(mam_event_mask, device=device, dtype=torch.bool)
        event_anchor_start = self._to_if_needed(mam_event_anchor_start, device=device, dtype=torch.long)
        event_anchor_end = self._to_if_needed(mam_event_anchor_end, device=device, dtype=torch.long)
        seg_start = self._to_if_needed(mam_seg_start, device=device, dtype=torch.long)
        seg_end = self._to_if_needed(mam_seg_end, device=device, dtype=torch.long)
        event_indices = self._to_if_needed(mam_event_indices, device=device, dtype=torch.long)
        event_cache_keys = self._to_if_needed(mam_event_cache_keys, device=device, dtype=torch.long)
        anchor_cache_keys = self._to_if_needed(mam_anchor_cache_keys, device=device, dtype=torch.long)
        if not event_mask.any():
            return (None, None)
        prompt_ids = list(getattr(self.config, 'mam_prompt_token_ids', []))
        cache_enabled = not self.training and event_cache_keys is not None and (event_cache_keys.ndim == 3) and (tuple(event_cache_keys.shape[:2]) == (batch_size, m_max))
        prefix_cache_key = None
        if cache_enabled:
            prefix_cache_key = self._mam_prefix_key_from_inputs(event_cache_keys=event_cache_keys, event_mask=event_mask, prompt_ids=prompt_ids, m_max=m_max, tokens_per_event=tokens_per_event)
            self._last_mam_prefix_cache_key = prefix_cache_key
            if prefix_cache_key is not None:
                cached_prefix = self._get_mam_prefix_cache().get(prefix_cache_key)
                if cached_prefix is not None:
                    cached_tokens, cached_mask = cached_prefix
                    return (self._to_if_needed(cached_tokens, device=device, dtype=encoder_dtype), self._to_if_needed(cached_mask, device=device, dtype=torch.bool))
        from airbrain.memory.mam.saliency import EventBudget
        events_bundle = EventBudget(event_indices=event_indices, event_mask=event_mask, num_events=event_mask.long().sum(dim=1), saliency=torch.zeros_like(history_actions[..., 0]), n_action=torch.zeros_like(history_actions[..., 0]), n_visual=torch.zeros_like(history_actions[..., 0]))
        qwen_model = self.qwen3_5_with_expert.vlm_backbone.model
        detach = bool(getattr(self.config, 'mam_detach_features', True))
        event_mask_list = event_mask.cpu().tolist()
        ev_anchor_start_list = event_anchor_start.cpu().tolist()
        ev_anchor_end_list = event_anchor_end.cpu().tolist()
        slots_to_encode: set[tuple[int, int]] = set()
        if cache_enabled:
            content_cache = self._get_mam_event_content_cache()
            event_key_rows = event_cache_keys.detach().cpu().tolist()
            expected_content_shape = (tokens_per_event, int(self.mam_memory_encoder.d_model))
            for b in range(batch_size):
                for k in range(m_max):
                    if not event_mask_list[b][k]:
                        continue
                    key = self._mam_event_key_from_row(event_key_rows[b][k])
                    cached = content_cache.get(key) if key is not None else None
                    if cached is None or tuple(cached.shape) != expected_content_shape:
                        slots_to_encode.add((b, k))
            if not slots_to_encode:
                result = self._encode_mam_memory_cached_only(history_actions=history_actions, events=events_bundle, seg_start=seg_start, seg_end=seg_end, mam_event_cache_keys=event_cache_keys)
            else:
                result = None
        else:
            result = None
            for b in range(batch_size):
                for k in range(m_max):
                    if event_mask_list[b][k]:
                        slots_to_encode.add((b, k))
        if result is None:
            anchor_n_cpu = anchor_n.cpu().tolist()
            anchor_offset = [0]
            for n in anchor_n_cpu:
                anchor_offset.append(anchor_offset[-1] + int(n))
            anchor_key_rows = None
            if anchor_cache_keys is not None:
                anchor_keys_cpu = anchor_cache_keys.detach().cpu()
                if anchor_keys_cpu.ndim == 1 and batch_size == 1:
                    anchor_key_rows = [anchor_keys_cpu.tolist()]
                elif anchor_keys_cpu.ndim == 2 and int(anchor_keys_cpu.shape[0]) == batch_size:
                    anchor_key_rows = anchor_keys_cpu.tolist()
            global_anchor_keys: list[Optional[int]] = []
            for b, n in enumerate(anchor_n_cpu):
                row = anchor_key_rows[b] if anchor_key_rows is not None and b < len(anchor_key_rows) else []
                for local_i in range(int(n)):
                    key = row[local_i] if local_i < len(row) else -1
                    global_anchor_keys.append(self._mam_anchor_key_from_value(key))
            required_global_indices: list[int] = []
            seen_required = set()
            for b, k in sorted(slots_to_encode):
                local_s = int(ev_anchor_start_list[b][k])
                local_e = int(ev_anchor_end_list[b][k])
                if local_s < 0 or local_e < 0:
                    continue
                base = anchor_offset[b]
                for global_idx in (base + local_s, base + local_e):
                    if global_idx not in seen_required:
                        seen_required.add(global_idx)
                        required_global_indices.append(global_idx)
            if not required_global_indices:
                return (None, None)
            feature_by_global: dict[int, torch.Tensor] = {}
            missing_global_indices: list[int] = []
            alias_global_indices: list[tuple[int, int]] = []
            pending_anchor_by_key: dict[int, int] = {}
            offline_reuse = getattr(self, '_offline_anchor_reuse', False)
            if offline_reuse and (self.training_objective != 'prior_warmup' or any((p.requires_grad for p in qwen_model.visual.parameters()))):
                raise ValueError('Offline anchor reuse requires frozen prior_warmup vision')
            anchor_feature_cache = self._get_mam_anchor_feature_cache() if not self.training or offline_reuse else {}
            for global_idx in required_global_indices:
                key = global_anchor_keys[global_idx] if global_idx < len(global_anchor_keys) else None
                cached = anchor_feature_cache.get(key) if key is not None else None
                if cached is not None and cached.ndim == 2 and (int(cached.shape[-1]) == self.hidden_size):
                    feature_by_global[global_idx] = cached.to(device=device, dtype=encoder_dtype)
                elif key is not None and key in pending_anchor_by_key:
                    alias_global_indices.append((global_idx, pending_anchor_by_key[key]))
                else:
                    missing_global_indices.append(global_idx)
                    if key is not None:
                        pending_anchor_by_key[key] = global_idx
            if missing_global_indices:
                grad_context = torch.no_grad() if detach else nullcontext()
                selected_pv, selected_grid = self._select_mam_anchor_inputs(anchor_pv=anchor_pv, anchor_grid=anchor_grid, anchor_indices=missing_global_indices)
                with grad_context:
                    image_outputs = qwen_model.get_image_features(selected_pv, selected_grid, return_dict=True)
                    flat_features = image_outputs.pooler_output
                if isinstance(flat_features, (list, tuple)):
                    selected_features = [self._to_if_needed(f, device=device) for f in flat_features]
                    if len(selected_features) != len(missing_global_indices):
                        selected_features = self._split_flat_mam_features(torch.cat(selected_features, dim=0), selected_grid, len(missing_global_indices))
                else:
                    selected_features = self._split_flat_mam_features(self._to_if_needed(flat_features, device=device), selected_grid, len(missing_global_indices))
                if detach:
                    selected_features = [f.detach() for f in selected_features]
                if encoder_dtype is not None:
                    selected_features = [f.to(dtype=encoder_dtype) for f in selected_features]
                for global_idx, feature in zip(missing_global_indices, selected_features, strict=False):
                    feature_by_global[global_idx] = feature
                    key = global_anchor_keys[global_idx] if global_idx < len(global_anchor_keys) else None
                    if key is not None:
                        anchor_feature_cache[key] = feature.detach().to('cpu')
            for global_idx, source_idx in alias_global_indices:
                feature_by_global[global_idx] = feature_by_global[source_idx]
            actual_patches = [int(f.shape[0]) for f in feature_by_global.values()]
            K_max = max(actual_patches) if actual_patches else 1
            K_max = max(K_max, 1)
            first_feature = next(iter(feature_by_global.values()), None)
            d_patch = first_feature.shape[-1] if first_feature is not None else self.hidden_size
            target_dtype = encoder_dtype or (first_feature.dtype if first_feature is not None else history_actions.dtype)
            endpoint_features = torch.zeros(batch_size, m_max, 2, K_max, d_patch, device=device, dtype=target_dtype)
            endpoint_patch_mask = torch.zeros(batch_size, m_max, 2, K_max, dtype=torch.bool, device=device)
            for b, k in sorted(slots_to_encode):
                local_s = int(ev_anchor_start_list[b][k])
                local_e = int(ev_anchor_end_list[b][k])
                if local_s < 0 or local_e < 0:
                    continue
                base = anchor_offset[b]
                f_s = feature_by_global[base + local_s]
                f_e = feature_by_global[base + local_e]
                Ls = min(int(f_s.shape[0]), K_max)
                Le = min(int(f_e.shape[0]), K_max)
                endpoint_features[b, k, 0, :Ls] = f_s[:Ls]
                endpoint_patch_mask[b, k, 0, :Ls] = True
                endpoint_features[b, k, 1, :Le] = f_e[:Le]
                endpoint_patch_mask[b, k, 1, :Le] = True
            if cache_enabled:
                result = self._encode_mam_memory_with_cache(history_actions=history_actions, events=events_bundle, seg_start=seg_start, seg_end=seg_end, endpoint_features=endpoint_features, endpoint_patch_mask=endpoint_patch_mask, mam_event_cache_keys=event_cache_keys)
            else:
                result = self.mam_memory_encoder.encode(history_actions=history_actions, history_mask=history_mask, events=events_bundle, seg_start=seg_start, seg_end=seg_end, endpoint_features=endpoint_features, endpoint_patch_mask=endpoint_patch_mask)
        tokens = result['tokens']
        mask = result['mask']
        m_max_tpe = m_max * tokens_per_event
        valid_counts = mask.long().sum(dim=1)
        batch_valid_max = int(valid_counts.max().item())
        if batch_valid_max == 0:
            return (None, None)
        if batch_valid_max < m_max_tpe:
            compact_tokens = tokens.new_zeros(batch_size, batch_valid_max, tokens.shape[-1])
            compact_mask = mask.new_zeros(batch_size, batch_valid_max)
            for b in range(batch_size):
                valid_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
                n_valid = int(valid_idx.numel())
                if n_valid > 0:
                    compact_tokens[b, :n_valid] = tokens[b, valid_idx]
                    compact_mask[b, :n_valid] = True
            tokens = compact_tokens
            mask = compact_mask
        if prompt_ids:
            prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            prompt_ids_t = prompt_ids_t.unsqueeze(0).expand(batch_size, -1)
            prompt_emb = qwen_model.get_input_embeddings()(prompt_ids_t)
            prompt_emb = prompt_emb.to(dtype=tokens.dtype, device=device)
            prompt_mask = torch.ones(batch_size, prompt_emb.shape[1], dtype=mask.dtype, device=device)
            tokens = torch.cat([prompt_emb, tokens], dim=1)
            mask = torch.cat([prompt_mask, mask], dim=1)
        if cache_enabled and prefix_cache_key is not None:
            self._get_mam_prefix_cache()[prefix_cache_key] = (tokens.detach().to('cpu'), mask.detach().to('cpu'))
        return (tokens, mask)

    def _prepend_vlm_memory_prefix(self, *, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], mm_token_type_ids: Optional[torch.IntTensor], action_attention_mask: Optional[torch.Tensor], memory_prefix_tokens: Optional[torch.Tensor], memory_prefix_mask: Optional[torch.Tensor], image_grid_thw: Optional[torch.LongTensor]=None, video_grid_thw: Optional[torch.LongTensor]=None, past_key_values=None) -> dict[str, Optional[torch.Tensor]]:
        if memory_prefix_tokens is None:
            return {'input_ids': input_ids, 'inputs_embeds': None, 'attention_mask': attention_mask, 'mm_token_type_ids': mm_token_type_ids, 'action_attention_mask': action_attention_mask, 'position_ids': None}
        batch_size, prefix_len, _ = memory_prefix_tokens.shape
        device = input_ids.device
        qwen_model = self.qwen3_5_with_expert.vlm_backbone.model
        input_embeddings = qwen_model.get_input_embeddings()
        token_embeds = input_embeddings(input_ids)
        prefix_embeds = self._to_if_needed(memory_prefix_tokens, device=device, dtype=token_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = self._to_if_needed(attention_mask, device=device)
        if memory_prefix_mask is None:
            memory_prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=torch.bool)
        else:
            memory_prefix_mask = self._to_if_needed(memory_prefix_mask, device=device, dtype=torch.bool)
        prefix_attention = self._to_if_needed(memory_prefix_mask, dtype=attention_mask.dtype)
        augmented_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        prefix_input_ids = torch.zeros(batch_size, prefix_len, device=device, dtype=input_ids.dtype)
        augmented_input_ids = torch.cat([prefix_input_ids, input_ids], dim=1)
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
            image_token_id = getattr(qwen_model.config, 'image_token_id', None)
            video_token_id = getattr(qwen_model.config, 'video_token_id', None)
            if image_token_id is not None:
                mm_token_type_ids[input_ids == int(image_token_id)] = 1
            if video_token_id is not None:
                mm_token_type_ids[input_ids == int(video_token_id)] = 2
        else:
            mm_token_type_ids = self._to_if_needed(mm_token_type_ids, device=device)
        prefix_mm_types = torch.zeros(batch_size, prefix_len, device=device, dtype=mm_token_type_ids.dtype)
        augmented_mm_token_type_ids = torch.cat([prefix_mm_types, mm_token_type_ids], dim=1)
        augmented_action_attention_mask = action_attention_mask
        if action_attention_mask is not None:
            augmented_action_attention_mask = torch.cat([self._to_if_needed(prefix_attention, device=action_attention_mask.device, dtype=action_attention_mask.dtype), action_attention_mask], dim=1)
        position_ids = None
        if hasattr(qwen_model, 'get_rope_index'):
            position_ids, rope_deltas = qwen_model.get_rope_index(augmented_input_ids, mm_token_type_ids=augmented_mm_token_type_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=augmented_attention_mask)
            qwen_model.rope_deltas = rope_deltas
        elif past_key_values is None:
            position_ids = augmented_attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(augmented_attention_mask == 0, 0)
        return {'input_ids': augmented_input_ids, 'inputs_embeds': inputs_embeds, 'attention_mask': augmented_attention_mask, 'mm_token_type_ids': augmented_mm_token_type_ids, 'action_attention_mask': augmented_action_attention_mask, 'position_ids': position_ids}

    def _prepare_vlm_prefix_embeds(self, *, input_ids: torch.LongTensor, inputs_embeds: Optional[torch.FloatTensor], attention_mask: Optional[torch.Tensor], position_ids: Optional[torch.LongTensor], past_key_values: Optional[Cache], pixel_values: Optional[torch.Tensor], pixel_values_videos: Optional[torch.FloatTensor], image_grid_thw: Optional[torch.LongTensor], video_grid_thw: Optional[torch.LongTensor], mm_token_type_ids: Optional[torch.IntTensor]) -> tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
        """Prepare VLM prefix embeddings for joint training.

        This mirrors the pre-language-model part of Qwen3_5Model.forward():
        token embedding lookup, image/video feature scatter, and M-RoPE
        position-id construction. The joint path then feeds these embeddings
        directly into the layer-by-layer VLM+Expert forward.
        """
        qwen_model = self.qwen3_5_with_expert.vlm_backbone.model
        if inputs_embeds is None:
            inputs_embeds = qwen_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_outputs = qwen_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_outputs = qwen_model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = qwen_model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        if position_ids is None:
            position_ids = qwen_model.compute_3d_position_ids(input_ids=input_ids, inputs_embeds=inputs_embeds, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask, past_key_values=past_key_values, mm_token_type_ids=mm_token_type_ids)
        return (inputs_embeds, position_ids)

    def _split_flat_mam_features(self, flat_features: torch.Tensor, image_grid_thw: Optional[torch.LongTensor], expected_images: int) -> list[torch.Tensor]:
        if expected_images <= 0:
            return []
        if flat_features.ndim == 3:
            if flat_features.shape[0] != expected_images:
                raise ValueError(f'MAM anchor feature/image count mismatch: expected {expected_images}, got {flat_features.shape[0]}.')
            return [flat_features[i] for i in range(expected_images)]
        if flat_features.ndim != 2:
            raise ValueError(f'Unsupported MAM anchor feature shape: {tuple(flat_features.shape)}.')
        if image_grid_thw is not None:
            counts = image_grid_thw.to(device=flat_features.device, dtype=torch.long).prod(dim=-1).tolist()
            if len(counts) == expected_images and sum(counts) == flat_features.shape[0]:
                return list(torch.split(flat_features, counts, dim=0))
        if flat_features.shape[0] % expected_images != 0:
            raise ValueError(f'Cannot split flat MAM anchor features without matching image_grid_thw: flat tokens={flat_features.shape[0]}, images={expected_images}.')
        tokens_per_image = flat_features.shape[0] // expected_images
        return list(flat_features.split(tokens_per_image, dim=0))

    def gradient_checkpointing_enable(self, **kwargs):
        self.qwen3_5_with_expert.vlm_backbone.gradient_checkpointing_enable(**kwargs)
        self.qwen3_5_with_expert.expert.gradient_checkpointing = True
        self.gradient_checkpointing_enabled = True

    def _training_source(self, states, actions, valid, history_hidden=None, motion_labels=None, motion_environment=None):
        if self.training_objective != 'expert_adapt':
            raise ValueError('Frozen template motion prior requires expert_adapt')
        source = self.motion_prior.sample(history_hidden, states, environment=motion_environment)
        return (None, source, None)

    def sample_noise(self, shape, device, generator=None):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device, generator=generator)

    def sample_time(self, bsize, device):
        if getattr(self.config, 'flow_time_distribution', 'beta') == 'uniform':
            return torch.rand((bsize,), device=device, dtype=torch.float32)
        beta_dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def _make_suffix_full_attention_mask(self, visible_prefix_mask: Optional[torch.Tensor], *, batch_size: int, suffix_len: int, prefix_key_value_length: int, device, dtype: torch.dtype) -> torch.Tensor:
        if visible_prefix_mask is None:
            sees_prefix = torch.zeros(batch_size, 1, suffix_len, prefix_key_value_length, device=device, dtype=dtype)
        else:
            visible_prefix_mask = visible_prefix_mask.to(device=device)
            if visible_prefix_mask.shape[1] < prefix_key_value_length:
                pad_len = prefix_key_value_length - visible_prefix_mask.shape[1]
                pad = torch.ones(visible_prefix_mask.shape[0], pad_len, device=device, dtype=visible_prefix_mask.dtype)
                visible_prefix_mask = torch.cat([visible_prefix_mask, pad], dim=1)
            elif visible_prefix_mask.shape[1] > prefix_key_value_length:
                visible_prefix_mask = visible_prefix_mask[:, :prefix_key_value_length]
            min_val = -10000.0
            sees_prefix = (~visible_prefix_mask.bool()).to(dtype) * min_val
            sees_prefix = sees_prefix[:, None, None, :].expand(batch_size, 1, suffix_len, prefix_key_value_length)
        suffix_full = torch.zeros(batch_size, 1, suffix_len, suffix_len, device=device, dtype=dtype)
        return torch.cat([sees_prefix, suffix_full], dim=-1)

    def _forward_action_step(self, *, input_ids: torch.LongTensor, attention_mask: torch.Tensor, position_ids: Optional[torch.LongTensor], states: torch.FloatTensor, actions: torch.FloatTensor, action_is_pad: Optional[torch.Tensor], action_attention_mask: Optional[torch.Tensor], prefix_hidden_states: torch.Tensor, prefix_past_key_values, prefix_rope_deltas: Optional[torch.Tensor], prefix_key_value_length: Optional[int]=None, visible_prefix_mask: Optional[torch.Tensor]=None, sample_valid_mask: Optional[torch.Tensor]=None, motion_labels: Optional[torch.Tensor]=None, motion_environment: Optional[torch.Tensor]=None) -> dict[str, Optional[torch.Tensor]]:
        actions_finite = torch.isfinite(actions).all(dim=-1)
        if action_is_pad is not None:
            actions_finite = actions_finite & ~action_is_pad.to(device=actions.device, dtype=torch.bool)
        if sample_valid_mask is not None:
            sample_valid_mask = sample_valid_mask.to(device=actions.device, dtype=torch.bool)
            actions_finite = actions_finite & sample_valid_mask[:, None]
        actions_clean = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions_for_model = torch.where(actions_finite.unsqueeze(-1), actions_clean, torch.zeros_like(actions_clean))
        prior_metrics = {}
        motion_hidden = self._history_hidden(prefix_hidden_states, input_ids, attention_mask)
        soft_labels, noise, prior_nll = self._training_source(states, actions_for_model, actions_finite, motion_hidden, motion_labels, motion_environment)
        flow_target = actions_for_model
        if prefix_past_key_values is None:
            raise RuntimeError('Prefix-first flow training requires prefix cache; disable gradient checkpointing')
        if prefix_key_value_length is None:
            prefix_key_value_length = prefix_past_key_values.get_seq_length()
        time = self.sample_time(actions_for_model.shape[0], actions.device)
        time_expanded = time[:, None, None].repeat(1, actions_for_model.shape[1], 1)
        x_t = time_expanded * noise + (1 - time_expanded) * flow_target
        u_t = noise - flow_target
        frozen_action_mask = None
        supervised_mask = actions_finite
        history_hidden = None
        suffix_embeds = self.embed_suffix(x_t, states, time, frozen_action_mask=frozen_action_mask, mode_ids=None)
        action_position_ids = self.get_action_position_ids(attention_mask, prefix_rope_deltas, position_ids=position_ids, suffix_len=suffix_embeds.shape[1])
        if visible_prefix_mask is None:
            visible_prefix_mask = action_attention_mask if action_attention_mask is not None else attention_mask
        expert_suffix_attention_mask = self._make_suffix_full_attention_mask(visible_prefix_mask, batch_size=input_ids.shape[0], suffix_len=suffix_embeds.shape[1], prefix_key_value_length=prefix_key_value_length, device=suffix_embeds.device, dtype=suffix_embeds.dtype)
        [_, suffix_hidden_states], _, _ = self.qwen3_5_with_expert.forward(suffix_embeds=suffix_embeds, attention_mask=expert_suffix_attention_mask, past_key_values=prefix_past_key_values, position_ids=action_position_ids, use_cache=True, full_suffix_attention=True, past_key_values_seq_length=prefix_key_value_length)
        router_probs = None
        action_time_embs = suffix_hidden_states[:, -self.config.chunk_size:]
        action_time_embs = action_time_embs.type(self.qwen3_5_with_expert.expert.action_out_proj.weight.dtype)
        v_t = self.qwen3_5_with_expert.expert.action_out_proj(action_time_embs)
        losses = (v_t.float() - u_t.float()).square()
        stop_weight = getattr(self, 'fm_stop_weight', 1.0)
        if stop_weight != 1.0:
            channel = torch.ones(losses.shape[-1], device=losses.device, dtype=losses.dtype)
            channel[-1] = stop_weight
            mask = supervised_mask.unsqueeze(-1).expand_as(losses)
            zeros = torch.zeros_like(losses)
            fm_loss = torch.where(mask, losses * channel, zeros).sum() / torch.where(mask, channel.expand_as(losses), zeros).sum().clamp_min(1e-06)
        else:
            valid_losses = losses.masked_select(supervised_mask.unsqueeze(-1).expand_as(losses))
            if valid_losses.numel() > 0:
                fm_loss = valid_losses.mean()
            else:
                fm_loss = torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        with torch.no_grad():
            step_mask = supervised_mask.unsqueeze(-1).float()
            denom = step_mask.sum().clamp_min(1.0)
            prior_metrics['fm_loss_xyz'] = (losses[..., :3].detach() * step_mask).sum() / (3.0 * denom)
            prior_metrics['fm_loss_stop'] = (losses[..., 3:].detach() * step_mask).sum() / denom
        loss = fm_loss
        if prior_nll is not None and self.training_objective in ('flow_joint', 'full_joint'):
            loss = loss + self.prior_loss_weight * prior_nll
        router_loss = None
        return {**prior_metrics, 'loss': loss, 'fm_loss': fm_loss, 'router_loss': router_loss, 'prior_nll': prior_nll, 'actions': v_t, 'noise': noise, 'suffix_hidden_states': suffix_hidden_states}

    def _add_dummy_loss(self, current_loss: Optional[torch.Tensor], parameters) -> Optional[torch.Tensor]:
        dummy = None
        for p in parameters:
            if not p.requires_grad:
                continue
            zero = p.reshape(-1)[0] * 0.0
            dummy = zero if dummy is None else dummy + zero
        if dummy is None:
            return current_loss
        return dummy if current_loss is None else current_loss + dummy

    def embed_suffix(self, noisy_actions: torch.Tensor, states: torch.Tensor, timestep: torch.Tensor, frozen_action_mask: Optional[torch.Tensor]=None, mode_ids: Optional[torch.Tensor]=None) -> torch.FloatTensor:
        expert = self.qwen3_5_with_expert.expert
        action_dtype = expert.action_in_proj.weight.dtype
        state_dtype = expert.state_proj.weight.dtype
        time_embs = create_sinusoidal_pos_embedding(timestep, dimension=expert.action_in_proj.out_features, device=timestep.device)
        noisy_actions = torch.nan_to_num(noisy_actions, nan=0.0, posinf=0.0, neginf=0.0)
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        noisy_actions = noisy_actions.to(dtype=action_dtype)
        states = states.to(dtype=state_dtype)
        dtype = action_dtype
        action_emb = expert.action_in_proj(noisy_actions).to(dtype=dtype)
        if action_emb.dim() == 2:
            action_emb = action_emb.unsqueeze(1)
        time_embs = time_embs[:, None, :].expand_as(action_emb).to(dtype=dtype)
        action_time_embs = torch.cat([action_emb, time_embs], dim=2)
        action_time_embs = expert.action_time_mlp_in(action_time_embs).to(dtype=dtype)
        action_time_embs = F.silu(action_time_embs)
        action_time_embs = expert.action_time_mlp_out(action_time_embs)
        if frozen_action_mask is not None and getattr(self, 'committed_action_embed', None) is not None:
            marker = self.committed_action_embed.to(device=action_time_embs.device, dtype=action_time_embs.dtype)
            action_time_embs = action_time_embs + frozen_action_mask.unsqueeze(-1).to(action_time_embs.dtype) * marker
        state_embs = expert.state_proj(states).to(dtype=action_time_embs.dtype)
        if state_embs.dim() == 2:
            state_embs = state_embs.unsqueeze(1)
        return torch.cat([state_embs, action_time_embs], dim=1)

    def get_action_position_ids(self, attention_mask: torch.Tensor, rope_deltas: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, suffix_len: Optional[int]=None) -> torch.LongTensor:
        """Compute 2D position IDs for action (suffix) tokens.

        Priority:
        1. If explicit 3D position_ids (from dataset/training) provided, read last
           temporal position directly from position_ids[0].
        2. If rope_deltas (from vlm_backbone) provided, compute as seq_len-1+delta.
        3. Fallback: seq_len - 1 (text-only, no visual position offset).

        Returns: [B, action_len]
        """
        B = attention_mask.shape[0]
        device = attention_mask.device
        token_indices = torch.arange(attention_mask.shape[1], device=device).view(1, -1)
        last_valid = torch.where(attention_mask.bool(), token_indices, torch.zeros_like(token_indices)).max(dim=1).values.long()
        if position_ids is not None:
            temporal = position_ids[0] if position_ids.dim() == 3 else position_ids
            last_ids = temporal.gather(1, last_valid.view(B, 1)).squeeze(1).long()
        elif rope_deltas is not None:
            delta = rope_deltas.squeeze(-1).to(device)
            last_ids = (last_valid + delta).long()
        else:
            last_ids = last_valid.long()
        action_len = suffix_len if suffix_len is not None else self.config.chunk_size + 1
        steps = torch.arange(1, action_len + 1, device=device).view(1, -1).expand(B, -1)
        return (last_ids.unsqueeze(1) + steps).long()

    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Cache]=None, inputs_embeds: Optional[torch.FloatTensor]=None, pixel_values: Optional[torch.Tensor]=None, pixel_values_videos: Optional[torch.FloatTensor]=None, image_grid_thw: Optional[torch.LongTensor]=None, video_grid_thw: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, cache_position: Optional[torch.LongTensor]=None, logits_to_keep: Union[int, torch.Tensor]=0, states: Optional[torch.FloatTensor]=None, actions: Optional[torch.FloatTensor]=None, time: Optional[torch.FloatTensor]=None, action_is_pad: torch.Tensor | None=None, action_attention_mask: Optional[torch.Tensor]=None, mm_token_type_ids: Optional[torch.IntTensor]=None, mam_history_actions: Optional[torch.Tensor]=None, mam_history_states: Optional[torch.Tensor]=None, mam_history_delta_states: Optional[torch.Tensor]=None, mam_history_mask: Optional[torch.Tensor]=None, mam_anchor_pixel_values: Optional[torch.Tensor]=None, mam_anchor_grid_thw: Optional[torch.LongTensor]=None, mam_anchor_n: Optional[torch.Tensor]=None, mam_event_indices: Optional[torch.Tensor]=None, mam_event_mask: Optional[torch.Tensor]=None, mam_seg_start: Optional[torch.Tensor]=None, mam_seg_end: Optional[torch.Tensor]=None, mam_event_anchor_start: Optional[torch.Tensor]=None, mam_event_anchor_end: Optional[torch.Tensor]=None, mam_event_cache_keys: Optional[torch.Tensor]=None, mam_anchor_cache_keys: Optional[torch.Tensor]=None, motion_labels: Optional[torch.Tensor]=None, motion_environment: Optional[torch.Tensor]=None, **kwargs) -> Tensor:
        if self.training and self.training_objective == 'legacy':
            raise ValueError('Configure an observation-conditioned source training objective before training')
        if self.training_objective not in ('legacy', 'full_joint') and (actions is None or (labels is not None and (labels != IGNORE_INDEX).any())):
            raise ValueError('Frozen-prefix stages support navigation-only batches without AR answers')
        use_cache = kwargs.pop('use_cache', None)
        if self.training:
            use_cache = False
        mam_prefix_tokens = None
        mam_prefix_mask = None
        if actions is not None:
            mam_prefix_tokens, mam_prefix_mask = self._make_mam_memory_prefix(batch_size=input_ids.shape[0], device=input_ids.device, mam_history_actions=mam_history_actions, mam_history_mask=mam_history_mask, mam_anchor_pixel_values=mam_anchor_pixel_values, mam_anchor_grid_thw=mam_anchor_grid_thw, mam_anchor_n=mam_anchor_n, mam_event_indices=mam_event_indices, mam_event_mask=mam_event_mask, mam_seg_start=mam_seg_start, mam_seg_end=mam_seg_end, mam_event_anchor_start=mam_event_anchor_start, mam_event_anchor_end=mam_event_anchor_end, mam_event_cache_keys=mam_event_cache_keys, mam_anchor_cache_keys=mam_anchor_cache_keys)
            prefix_inputs = self._prepend_vlm_memory_prefix(input_ids=input_ids, attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids, action_attention_mask=action_attention_mask, memory_prefix_tokens=mam_prefix_tokens, memory_prefix_mask=mam_prefix_mask, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw)
            prefix_position_ids = prefix_inputs['position_ids'] if prefix_inputs['position_ids'] is not None else position_ids
            with torch.no_grad() if self._prefix_forward_no_grad() else nullcontext():
                [prefix_hidden_states, _], prefix_past_key_values, prefix_rope_deltas = self.qwen3_5_with_expert.forward(input_ids=prefix_inputs['input_ids'], inputs_embeds=prefix_inputs['inputs_embeds'], attention_mask=prefix_inputs['attention_mask'], position_ids=prefix_position_ids, past_key_values=past_key_values, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, mm_token_type_ids=prefix_inputs['mm_token_type_ids'], cache_position=cache_position, use_cache=self.training_objective != 'prior_warmup')
            step_result = self._forward_action_step(motion_labels=motion_labels, motion_environment=motion_environment, input_ids=prefix_inputs['input_ids'], attention_mask=prefix_inputs['attention_mask'], position_ids=prefix_position_ids, states=states, actions=actions, action_is_pad=action_is_pad, action_attention_mask=prefix_inputs['action_attention_mask'], prefix_hidden_states=prefix_hidden_states, prefix_past_key_values=prefix_past_key_values, prefix_rope_deltas=prefix_rope_deltas)
        else:
            [prefix_hidden_states, _], _, _ = self.qwen3_5_with_expert.forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, mm_token_type_ids=mm_token_type_ids, cache_position=cache_position, use_cache=use_cache)
        loss = None
        fm_loss = None
        router_loss = None
        prior_nll = None
        logits = None
        v_t = None
        noise = None
        if actions is not None:
            loss = step_result['loss']
            fm_loss = step_result['fm_loss']
            router_loss = step_result['router_loss']
            prior_nll = step_result['prior_nll']
            v_t = step_result['actions']
            noise = step_result['noise']
        ar_loss = None
        has_ar_labels = labels is not None and bool((labels != IGNORE_INDEX).any().item())
        if has_ar_labels:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.qwen3_5_with_expert.vlm_backbone.lm_head(prefix_hidden_states[:, slice_indices, :])
            ar_loss = self.qwen3_5_with_expert.vlm_backbone.loss_function(logits=logits, labels=labels, vocab_size=self.config.vlm_config.text_config.vocab_size, **kwargs)
            loss = loss + ar_loss if loss is not None else ar_loss
        if actions is None:
            loss = self._add_dummy_loss(loss, self.qwen3_5_with_expert.expert.parameters())
            if getattr(self, 'committed_action_embed', None) is not None:
                loss = self._add_dummy_loss(loss, [self.committed_action_embed])
            if getattr(self, 'mode_embedding', None) is not None:
                loss = self._add_dummy_loss(loss, None.parameters())
            loss = self._add_dummy_loss(loss, self.motion_prior.parameters())
        if not has_ar_labels:
            loss = self._add_dummy_loss(loss, self.qwen3_5_with_expert.vlm_backbone.lm_head.parameters())
            loss = self._add_dummy_loss(loss, self.qwen3_5_with_expert.vlm_backbone.model.language_model.parameters())
        if self.mam_memory_encoder is not None:
            mam_was_active = actions is not None and mam_prefix_tokens is not None
            if not mam_was_active:
                loss = self._add_dummy_loss(loss, self.mam_memory_encoder.parameters())
        visual = getattr(getattr(self.qwen3_5_with_expert.vlm_backbone, 'model', None), 'visual', None)
        if visual is not None and pixel_values is None and (pixel_values_videos is None) and (mam_anchor_pixel_values is None):
            loss = self._add_dummy_loss(loss, visual.parameters())
        return OpenFLYOutputWithPast(loss=loss, fm_loss=fm_loss.detach() if fm_loss is not None else None, ar_loss=ar_loss.detach() if ar_loss is not None else None, router_loss=router_loss.detach() if router_loss is not None else None, prior_nll=prior_nll.detach() if prior_nll is not None else None, prior_mean_mse=step_result.get('prior_mean_mse', None).detach() if actions is not None and step_result.get('prior_mean_mse') is not None else None, prior_std_mean=step_result.get('prior_std_mean', None).detach() if actions is not None and step_result.get('prior_std_mean') is not None else None, fm_loss_xyz=step_result.get('fm_loss_xyz') if actions is not None else None, fm_loss_stop=step_result.get('fm_loss_stop') if actions is not None else None, prior_std_floor_frac=step_result.get('prior_std_floor_frac') if actions is not None else None, prior_z_absmax=step_result.get('prior_z_absmax') if actions is not None else None, actions=v_t.detach() if v_t is not None else None, noise=noise.detach() if noise is not None else None, logits=None, past_key_values=None, rope_deltas=None)

    @torch.no_grad()
    def sample_actions(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None, pixel_values_videos: Optional[torch.FloatTensor]=None, image_grid_thw: Optional[torch.LongTensor]=None, states: Optional[torch.FloatTensor]=None, denoise_steps: int=5, mm_token_type_ids: Optional[torch.IntTensor]=None, mam_history_actions: Optional[torch.Tensor]=None, mam_history_states: Optional[torch.Tensor]=None, mam_history_delta_states: Optional[torch.Tensor]=None, mam_history_mask: Optional[torch.Tensor]=None, mam_anchor_pixel_values: Optional[torch.Tensor]=None, mam_anchor_grid_thw: Optional[torch.LongTensor]=None, mam_anchor_n: Optional[torch.Tensor]=None, mam_event_indices: Optional[torch.Tensor]=None, mam_event_mask: Optional[torch.Tensor]=None, mam_seg_start: Optional[torch.Tensor]=None, mam_seg_end: Optional[torch.Tensor]=None, mam_event_anchor_start: Optional[torch.Tensor]=None, mam_event_anchor_end: Optional[torch.Tensor]=None, mam_event_cache_keys: Optional[torch.Tensor]=None, mam_anchor_cache_keys: Optional[torch.Tensor]=None, frozen_actions: Optional[torch.Tensor]=None, prev_action: Optional[torch.Tensor]=None, gmm_init_mode: str='weighted_mean', gmm_generator=None, motion_environment: Optional[torch.Tensor]=None, **kwargs) -> Tensor:
        device = next(self.parameters()).device
        mam_prefix_tokens, mam_prefix_mask = self._make_mam_memory_prefix(batch_size=input_ids.shape[0], device=device, mam_history_actions=mam_history_actions, mam_history_mask=mam_history_mask, mam_anchor_pixel_values=mam_anchor_pixel_values, mam_anchor_grid_thw=mam_anchor_grid_thw, mam_anchor_n=mam_anchor_n, mam_event_indices=mam_event_indices, mam_event_mask=mam_event_mask, mam_seg_start=mam_seg_start, mam_seg_end=mam_seg_end, mam_event_anchor_start=mam_event_anchor_start, mam_event_anchor_end=mam_event_anchor_end, mam_event_cache_keys=mam_event_cache_keys, mam_anchor_cache_keys=mam_anchor_cache_keys)
        prefix_inputs = self._prepend_vlm_memory_prefix(input_ids=input_ids, attention_mask=attention_mask, mm_token_type_ids=mm_token_type_ids, action_attention_mask=None, memory_prefix_tokens=mam_prefix_tokens, memory_prefix_mask=mam_prefix_mask, image_grid_thw=image_grid_thw, video_grid_thw=None)
        self._warn_if_padded_shared_linear_prefix(prefix_inputs['attention_mask'])
        prefix_position_ids = prefix_inputs['position_ids'] if prefix_inputs['position_ids'] is not None else None
        mam_prefix_cache_key = getattr(self, '_last_mam_prefix_cache_key', None)
        vlm_prefix_cache_size = int(getattr(self.config, 'runtime_vlm_prefix_cache_size', 0))
        vlm_prefix_cache_key = None
        if vlm_prefix_cache_size > 0 and (mam_prefix_tokens is None or mam_prefix_cache_key is not None):
            vlm_prefix_cache_key = self._vlm_prefix_cache_key(input_ids=input_ids, attention_mask=attention_mask, position_ids=None, mam_prefix_cache_key=mam_prefix_cache_key, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, video_grid_thw=None, mm_token_type_ids=mm_token_type_ids)
        cached_vlm_prefix = self._get_runtime_vlm_prefix_cache().get(vlm_prefix_cache_key) if vlm_prefix_cache_key is not None else None
        if cached_vlm_prefix is not None:
            cached_hidden, cached_past_key_values, cached_rope_deltas = cached_vlm_prefix
            prefix_hidden_states = self._to_if_needed(cached_hidden, device=device)
            past_key_values = self._clone_cache_for_reuse(cached_past_key_values)
            prefix_rope_deltas = self._to_if_needed(cached_rope_deltas, device=device)
        else:
            [prefix_hidden_states, _], past_key_values, prefix_rope_deltas = self.qwen3_5_with_expert.forward(input_ids=prefix_inputs['input_ids'], inputs_embeds=prefix_inputs['inputs_embeds'], attention_mask=prefix_inputs['attention_mask'], position_ids=prefix_position_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw, mm_token_type_ids=prefix_inputs['mm_token_type_ids'], use_cache=True)
            if vlm_prefix_cache_key is not None:
                self._store_runtime_vlm_prefix_cache(vlm_prefix_cache_key, (prefix_hidden_states.detach(), self._clone_cache_for_reuse(past_key_values), prefix_rope_deltas.detach() if isinstance(prefix_rope_deltas, torch.Tensor) else None))
        prefix_len = past_key_values.get_seq_length()
        B = input_ids.shape[0]
        input_ids = prefix_inputs['input_ids']
        attention_mask = prefix_inputs['attention_mask']
        if attention_mask is None:
            attention_mask = torch.ones(B, input_ids.shape[1], device=device, dtype=torch.long)
        history_hidden = None
        whiten_params = None
        if getattr(self, '_eval_prev_action_prior', None) is not None:
            raise ValueError('motion_gmm cannot be combined with the inference-only previous-action prior')
        h = self._history_hidden(prefix_hidden_states, input_ids, attention_mask)
        weighted_noise = self.motion_prior.sample(h, states, generator=gmm_generator, environment=motion_environment)
        if not torch.isfinite(weighted_noise).all():
            raise RuntimeError('NaN/Inf detected in initial action noise during sample_actions.')
        x_t = weighted_noise.type(self.qwen3_5_with_expert.expert.action_in_proj.weight.dtype)
        L = x_t.shape[1]
        frozen_action_mask = None
        frozen_prefix_full = None
        if frozen_actions is not None and self._action_prefix_enabled():
            f = max(0, min(int(frozen_actions.shape[1]), L))
            if f > 0:
                frozen_action_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
                frozen_action_mask[:, :f] = True
                frozen_prefix_full = x_t.clone()
                committed = frozen_actions[:, :f].to(device=device, dtype=torch.float32)
                frozen_prefix_full[:, :f] = committed.to(x_t.dtype)
                x_t = torch.where(frozen_action_mask.unsqueeze(-1), frozen_prefix_full, x_t)
        actual_suffix_len = x_t.shape[1] + 1
        action_position_ids = self.get_action_position_ids(attention_mask, prefix_rope_deltas, position_ids=prefix_position_ids, suffix_len=actual_suffix_len)
        expert_suffix_attention_mask = None
        expert_suffix_attention_mask = self._make_suffix_full_attention_mask(attention_mask, batch_size=B, suffix_len=actual_suffix_len, prefix_key_value_length=prefix_len, device=x_t.device, dtype=x_t.dtype)
        dt = -1.0 / denoise_steps
        for step in range(denoise_steps):
            time = torch.full((B,), 1.0 + step * dt, device=device, dtype=torch.float32)
            suffix_embeds = self.embed_suffix(x_t, states, time, frozen_action_mask=frozen_action_mask, mode_ids=None)
            outputs, _, _ = self.qwen3_5_with_expert.forward(suffix_embeds=suffix_embeds, attention_mask=expert_suffix_attention_mask, past_key_values=past_key_values, position_ids=action_position_ids, use_cache=True, full_suffix_attention=True, past_key_values_seq_length=prefix_len)
            suffix_hidden_states = outputs[1]
            action_time_embs = suffix_hidden_states[:, -L:]
            v_t = self.qwen3_5_with_expert.expert.action_out_proj(action_time_embs).to(x_t.dtype)
            if not torch.isfinite(v_t).all():
                raise RuntimeError(f'NaN/Inf detected in action velocity at denoise step {step}.')
            x_t = x_t + dt * v_t
            if frozen_action_mask is not None:
                x_t = torch.where(frozen_action_mask.unsqueeze(-1), frozen_prefix_full, x_t)
            if not torch.isfinite(x_t).all():
                raise RuntimeError(f'NaN/Inf detected in sampled actions at denoise step {step}.')
            past_key_values.crop(prefix_len)
        return (x_t, weighted_noise)
