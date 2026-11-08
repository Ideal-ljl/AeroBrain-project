"""Top-level Motion-Adaptive Memory encoder (two-stage interface).

Stage 1 — event selection (action-only, no vision tower):
  saliency(actions, mask) → detect_events → events + per-event segment endpoints

Stage 2 — encoding (after the caller has run vision tower on selected anchors only):
  encode(actions, mask, events, seg_endpoints, endpoint_patches) → memory tokens

The two-stage split lets callers run the vision tower on the ~M_max selected
anchor frames instead of all N history frames, which is the main GPU efficiency
win for chunk-stride training.

Output:
  tokens : [B, M_max × tokens_per_event, d_model]
  mask   : [B, M_max × tokens_per_event]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from airbrain.memory.mam.saliency import compute_saliency, detect_events, EventBudget
from airbrain.memory.mam.segment_encoder import (
    ActionSegmentEncoder,
    SegmentCrossAttention,
)


class MAMMemoryEncoder(nn.Module):
    """Memory encoder that emits VLM-prefix tokens via motion-adaptive segmentation."""

    def __init__(
        self,
        *,
        action_dim: int,
        d_model: int,
        patch_dim: int | None = None,
        chunk_size: int = 12,
        history_steps: int = 48,
        m_max: int = 16,
        tokens_per_event: int = 2,
        nms_gap: int = -1,
        tau_percentile: float = 80.0,
        lambda_visual: float = 1.0,   # retained for backwards compat; ignored by select_events
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        detach_features: bool = True,
        max_age: int = 256,
        chunk_boundary_mode: str = "first_only",
    ) -> None:
        super().__init__()
        if d_model <= 0 or action_dim <= 0:
            raise ValueError("d_model and action_dim must be positive")
        if m_max <= 0 or tokens_per_event <= 0:
            raise ValueError("m_max and tokens_per_event must be positive")

        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.patch_dim = int(patch_dim or d_model)
        self.chunk_size = int(chunk_size)
        self.history_steps = int(history_steps)
        self.m_max = int(m_max)
        self.tokens_per_event = int(tokens_per_event)
        self.nms_gap = int(nms_gap) if nms_gap >= 0 else max(1, self.chunk_size // 2)
        self.tau_percentile = float(tau_percentile)
        self.lambda_visual = float(lambda_visual)
        self.detach_features = bool(detach_features)
        self.max_age = int(max_age)
        if chunk_boundary_mode not in ("all", "first_only", "none"):
            raise ValueError(
                f"chunk_boundary_mode must be 'all'/'first_only'/'none', got {chunk_boundary_mode!r}"
            )
        self.chunk_boundary_mode = str(chunk_boundary_mode)

        self.action_encoder = ActionSegmentEncoder(
            action_dim=self.action_dim,
            d_model=self.d_model,
            tokens_per_event=self.tokens_per_event,
            max_age=self.max_age,
        )
        self.cross_attn = SegmentCrossAttention(
            d_model=self.d_model,
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            dropout=float(dropout),
            patch_proj_dim=self.patch_dim,
        )

        # Trainable embedding to mark empty / no-event slots so the
        # cross-attention has something legal to attend to.
        self.empty_kv_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.empty_kv_token, std=0.02)

    # ------------------------------------------------------------------
    # Stage 1: event selection (action-only, no vision features needed)
    # ------------------------------------------------------------------

    def select_events(
        self,
        history_actions: torch.Tensor,    # [B, T, A]
        history_mask: torch.Tensor,       # [B, T]
    ) -> tuple[EventBudget, torch.Tensor, torch.Tensor]:
        """Detect events from actions alone.

        Returns
        -------
        events : EventBudget
            event_indices [B, M_max], event_mask [B, M_max], num_events [B]
        seg_start : [B, M_max] step index of segment start (previous event step,
                    or first valid step for the first event).
        seg_end :   [B, M_max] step index of segment end (= event index).
        """
        if history_actions.ndim != 3:
            raise ValueError(f"history_actions must be [B, T, A], got {tuple(history_actions.shape)}")
        B, T, A = history_actions.shape
        if A != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {A}")
        if history_mask.shape != (B, T):
            raise ValueError(f"history_mask shape mismatch: expected {(B, T)}, got {tuple(history_mask.shape)}")
        device = history_actions.device

        sal_info = compute_saliency(
            history_actions,
            pooled_visual=None,           # action-only — no vision tower required yet
            mask=history_mask,
            lambda_visual=0.0,
        )
        events = detect_events(
            sal_info["saliency"],
            mask=history_mask,
            chunk_size=self.chunk_size,
            m_max=self.m_max,
            nms_gap=self.nms_gap,
            tau_percentile=self.tau_percentile,
            chunk_boundary_mode=self.chunk_boundary_mode,
        )

        # Prefetch to CPU once to avoid B × M_max GPU syncs in the loop body.
        first_valid_cpu = history_mask.long().argmax(dim=1).cpu().tolist()
        event_mask_cpu = events.event_mask.cpu().tolist()
        event_indices_cpu = events.event_indices.cpu().tolist()

        seg_start_list = []
        seg_end_list = []
        for b in range(B):
            prev = int(first_valid_cpu[b])
            row_s = []
            row_e = []
            for k in range(self.m_max):
                if not event_mask_cpu[b][k]:
                    row_s.append(0)
                    row_e.append(0)
                    continue
                ev = int(event_indices_cpu[b][k])
                row_s.append(prev)
                row_e.append(ev)
                prev = ev
            seg_start_list.append(row_s)
            seg_end_list.append(row_e)
        seg_start = torch.tensor(seg_start_list, dtype=torch.long, device=device)
        seg_end = torch.tensor(seg_end_list, dtype=torch.long, device=device)
        return events, seg_start, seg_end

    # ------------------------------------------------------------------
    # Stage 2: cross-attention encoding with pre-gathered anchor features
    # ------------------------------------------------------------------

    def build_event_segments(
        self,
        *,
        history_actions: torch.Tensor,           # [B, T, A]
        event_mask: torch.Tensor,                # [B, M_max]
        seg_start: torch.Tensor,                 # [B, M_max]
        seg_end: torch.Tensor,                   # [B, M_max]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather per-event action segments and runtime ages."""
        if history_actions.ndim != 3:
            raise ValueError("history_actions must be [B, T, A]")
        B, T, _ = history_actions.shape
        device = history_actions.device

        if event_mask.shape != (B, self.m_max):
            raise ValueError(
                f"event_mask shape mismatch: expected {(B, self.m_max)}, got {tuple(event_mask.shape)}"
            )
        if seg_start.shape != (B, self.m_max) or seg_end.shape != (B, self.m_max):
            raise ValueError(
                f"seg_start/seg_end must be [B, m_max]={B, self.m_max}, "
                f"got {tuple(seg_start.shape)} / {tuple(seg_end.shape)}"
            )

        action_dim = self.action_dim
        L_max = T

        # Single sync: pull all needed indices to CPU and build host-side lists.
        event_mask_cpu = event_mask.cpu().tolist()
        seg_start_cpu = seg_start.cpu().tolist()
        seg_end_cpu = seg_end.cpu().tolist()
        n_evt_per_sample = [sum(row) for row in event_mask_cpu]

        seg_action_pad = history_actions.new_zeros(B, self.m_max, L_max, action_dim)
        seg_lengths_list = [[0] * self.m_max for _ in range(B)]
        ages_list = [[0] * self.m_max for _ in range(B)]

        for b in range(B):
            n_evt = n_evt_per_sample[b]
            ha_b = history_actions[b]  # [T, A] — single tensor view, no per-step sync
            for k in range(self.m_max):
                if not event_mask_cpu[b][k]:
                    continue
                s = seg_start_cpu[b][k]
                e = seg_end_cpu[b][k]
                if e < s:
                    e = s
                length = min(e - s + 1, L_max)
                seg_action_pad[b, k, :length] = ha_b[s : s + length]
                seg_lengths_list[b][k] = length
                ages_list[b][k] = (n_evt - 1) - k

        seg_lengths = torch.tensor(seg_lengths_list, dtype=torch.long, device=device)
        ages = torch.tensor(ages_list, dtype=torch.long, device=device)
        return seg_action_pad, seg_lengths, ages

    def encode_event_content(
        self,
        *,
        segment_actions: torch.Tensor,          # [N, L_max, A]
        segment_lengths: torch.Tensor,          # [N]
        endpoint_features: torch.Tensor,        # [N, 2, K_max, d_patch]
        endpoint_patch_mask: torch.Tensor,      # [N, 2, K_max] True = valid
    ) -> torch.Tensor:
        """Encode age-independent event content tokens.

        The returned tokens intentionally do not include age/recency embedding,
        so inference can cache them for a fixed event segment and add age at use
        time.
        """
        if segment_actions.ndim != 3:
            raise ValueError("segment_actions must be [N, L, A]")
        N, _, action_dim = segment_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim={self.action_dim}, got {action_dim}")
        if segment_lengths.shape != (N,):
            raise ValueError(
                f"segment_lengths shape mismatch: expected {(N,)}, got {tuple(segment_lengths.shape)}"
            )
        if endpoint_features.ndim != 4:
            raise ValueError(
                f"endpoint_features must be [N, 2, K, d_patch], got {tuple(endpoint_features.shape)}"
            )
        if endpoint_features.shape[0] != N or endpoint_features.shape[1] != 2:
            raise ValueError(
                f"endpoint_features shape mismatch: expected [N={N}, 2, K, d], "
                f"got {tuple(endpoint_features.shape)}"
            )
        if endpoint_patch_mask.shape != endpoint_features.shape[:3]:
            raise ValueError(
                f"endpoint_patch_mask shape mismatch: expected {tuple(endpoint_features.shape[:3])}, "
                f"got {tuple(endpoint_patch_mask.shape)}"
            )

        device = segment_actions.device
        K_max = endpoint_features.shape[2]
        d_patch = endpoint_features.shape[3]
        kv_buf = endpoint_features.reshape(N, 2 * K_max, d_patch)
        kv_pad_mask = (~endpoint_patch_mask.reshape(N, 2 * K_max)).contiguous()

        if self.detach_features:
            kv_buf = kv_buf.detach()

        # A cache miss should normally correspond to a real event with real
        # anchors. Keep the empty-slot fallback here so the public helper remains
        # legal for padded slots and mirrors encode().
        empty_slots = (~endpoint_patch_mask.reshape(N, 2 * K_max).any(dim=1))
        if empty_slots.any():
            slot_idx = empty_slots.nonzero(as_tuple=False).flatten()
            empty_tok = self.empty_kv_token.to(dtype=kv_buf.dtype, device=device).view(-1)
            d_use = min(kv_buf.shape[-1], empty_tok.numel())
            # `reshape` (and `detach`) return views sharing storage with the
            # caller's `endpoint_features`, so writing the empty-slot token in
            # place would mutate the caller's tensor. Clone first.
            kv_buf = kv_buf.clone()
            for row in slot_idx.tolist():
                kv_buf[row, 0, :d_use] = empty_tok[:d_use]
                kv_pad_mask[row, 0] = False

        dummy_ages = torch.zeros_like(segment_lengths)
        queries = self.action_encoder(segment_actions, segment_lengths, dummy_ages)
        return self.cross_attn(queries, kv_buf, kv_padding_mask=kv_pad_mask)

    def encode(
        self,
        *,
        history_actions: torch.Tensor,           # [B, T, A]
        history_mask: torch.Tensor,              # [B, T]
        events: EventBudget,                     # from select_events
        seg_start: torch.Tensor,                 # [B, M_max]
        seg_end: torch.Tensor,                   # [B, M_max]
        endpoint_features: torch.Tensor,         # [B, M_max, 2, K_max, d_patch]
        endpoint_patch_mask: torch.Tensor,       # [B, M_max, 2, K_max] True = valid
    ) -> dict[str, torch.Tensor]:
        """Cross-attention given pre-gathered start/end frame patches.

        endpoint_features[b, k, 0, :, :] = patches of frame at seg_start[b, k]
        endpoint_features[b, k, 1, :, :] = patches of frame at seg_end[b, k]
        """
        if history_actions.ndim != 3:
            raise ValueError("history_actions must be [B, T, A]")
        B, T, _ = history_actions.shape
        event_mask = events.event_mask
        event_indices = events.event_indices
        seg_action_pad, seg_lengths, ages = self.build_event_segments(
            history_actions=history_actions,
            event_mask=event_mask,
            seg_start=seg_start,
            seg_end=seg_end,
        )

        # --- build KV buffer from endpoint features ---
        # endpoint_features layout: [B, m_max, 2, K_max, d_patch]
        # Flatten 2 endpoints into a single sequence per event → [B, m_max, 2*K_max, d_patch]
        if endpoint_features.ndim != 5:
            raise ValueError(
                f"endpoint_features must be [B, m_max, 2, K_max, d_patch], "
                f"got {tuple(endpoint_features.shape)}"
            )
        if endpoint_features.shape[0] != B or endpoint_features.shape[1] != self.m_max:
            raise ValueError(
                f"endpoint_features shape mismatch: expected [B={B}, m_max={self.m_max}, 2, K, d], "
                f"got {tuple(endpoint_features.shape)}"
            )
        seg_actions_flat = seg_action_pad.reshape(B * self.m_max, T, self.action_dim)
        seg_lengths_flat = seg_lengths.reshape(B * self.m_max)
        ages_flat = ages.reshape(B * self.m_max)

        # --- cross-attention ---
        content_tokens = self.encode_event_content(
            segment_actions=seg_actions_flat,
            segment_lengths=seg_lengths_flat,
            endpoint_features=endpoint_features.reshape(
                B * self.m_max, 2, endpoint_features.shape[3], endpoint_features.shape[4]
            ),
            endpoint_patch_mask=endpoint_patch_mask.reshape(
                B * self.m_max, 2, endpoint_features.shape[3]
            ),
        )
        # [B*m_max, tokens_per_event, d_model]; independent of age/recency and
        # therefore cacheable for a fixed event segment + visual endpoints.
        age_tokens = self.action_encoder.age_tokens(ages_flat, dtype=content_tokens.dtype)
        out = content_tokens + age_tokens

        content_memory_tokens = content_tokens.reshape(
            B, self.m_max, self.tokens_per_event, self.d_model
        )
        content_memory_tokens = content_memory_tokens.reshape(
            B, self.m_max * self.tokens_per_event, self.d_model
        )
        memory_tokens = out.reshape(B, self.m_max, self.tokens_per_event, self.d_model)
        memory_tokens = memory_tokens.reshape(B, self.m_max * self.tokens_per_event, self.d_model)

        token_mask = event_mask.unsqueeze(-1).expand(-1, -1, self.tokens_per_event)
        token_mask = token_mask.reshape(B, self.m_max * self.tokens_per_event)
        memory_tokens = memory_tokens * token_mask.unsqueeze(-1).to(memory_tokens.dtype)
        content_memory_tokens = content_memory_tokens * token_mask.unsqueeze(-1).to(content_memory_tokens.dtype)

        return {
            "tokens": memory_tokens,
            "content_tokens": content_memory_tokens,
            "mask": token_mask,
            "num_events": events.num_events,
            "event_indices": event_indices,
        }
