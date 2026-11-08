"""Action saliency and event selection for motion-adaptive memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EventBudget:
    """Result of per-sample event detection.

    All tensors are batched and right-padded to a fixed cap.
    """

    event_indices: torch.LongTensor
    event_mask: torch.BoolTensor
    num_events: torch.LongTensor
    saliency: torch.Tensor
    n_action: torch.Tensor
    n_visual: torch.Tensor


def _robust_scale_batched(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-row median of non-zero entries (fully vectorized).

    `values` shape `[B, T]`. Returns `[B]`. Falls back to 1.0 for rows whose
    non-zero count is zero.
    """
    masked = values.where(values > eps, torch.full_like(values, float("nan")))
    medians = torch.nanmedian(masked, dim=1).values
    medians = torch.nan_to_num(medians, nan=1.0)
    return medians.clamp_min(1e-8)


def compute_action_novelty(
    actions: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """`||a_i - a_{i-1}||` along the time axis. `actions` shape `[B, T, A]`.

    Pad slots (where mask is False at either i or i-1) emit 0 novelty.
    Returns `[B, T]` with index 0 always zero.
    """
    B, T, _ = actions.shape
    out = actions.new_zeros((B, T))
    if T < 2:
        return out
    diffs = torch.norm(actions[:, 1:] - actions[:, :-1], dim=-1)
    if mask is not None:
        valid = mask[:, 1:].bool() & mask[:, :-1].bool()
        diffs = diffs * valid.to(diffs.dtype)
    out[:, 1:] = diffs
    return out


def compute_visual_novelty(
    pooled_features: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """`1 - cos(pool(f_i), pool(f_{i-1}))`. `pooled_features` shape `[B, T, d]`.

    Returns `[B, T]` with index 0 always zero. Pad steps emit 0.
    """
    B, T, _ = pooled_features.shape
    out = pooled_features.new_zeros((B, T))
    if T < 2:
        return out
    normed = torch.nn.functional.normalize(pooled_features, dim=-1)
    cos = (normed[:, 1:] * normed[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
    diffs = 1.0 - cos
    if mask is not None:
        valid = mask[:, 1:].bool() & mask[:, :-1].bool()
        diffs = diffs * valid.to(diffs.dtype)
    out[:, 1:] = diffs
    return out


def compute_saliency(
    actions: torch.Tensor,
    *,
    pooled_visual: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    lambda_visual: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute per-step saliency `s = max(n_action_z, λ · n_visual_z)`.

    Args:
        actions: `[B, T, A]` normalized actions.
        pooled_visual: optional `[B, T, d]` pooled vision-tower features.
        mask: optional `[B, T]` validity mask.
        lambda_visual: weight on visual novelty after z-score normalization.
    """
    n_action_raw = compute_action_novelty(actions, mask=mask)
    sigma_a = _robust_scale_batched(n_action_raw)
    n_action = n_action_raw / sigma_a.unsqueeze(1)

    if pooled_visual is not None:
        n_visual_raw = compute_visual_novelty(pooled_visual, mask=mask)
        sigma_v = _robust_scale_batched(n_visual_raw)
        n_visual = n_visual_raw / sigma_v.unsqueeze(1)
    else:
        n_visual = torch.zeros_like(n_action)
        sigma_v = n_action.new_ones(n_action.shape[0])

    saliency = torch.maximum(n_action, lambda_visual * n_visual)
    if mask is not None:
        saliency = saliency * mask.to(saliency.dtype)
    return {
        "saliency": saliency,
        "n_action": n_action,
        "n_visual": n_visual,
        "sigma_a": sigma_a,
        "sigma_v": sigma_v,
    }


def _nms_1d_single(
    saliency_row: torch.Tensor, candidates: list[int], min_gap: int
) -> list[int]:
    if min_gap <= 0 or not candidates:
        return sorted(candidates)
    ranked = sorted(candidates, key=lambda i: -float(saliency_row[i]))
    kept: list[int] = []
    for i in ranked:
        if all(abs(i - k) >= min_gap for k in kept):
            kept.append(i)
    return sorted(kept)


def detect_events(
    saliency: torch.Tensor,
    *,
    mask: torch.Tensor,
    chunk_size: int,
    m_max: int,
    nms_gap: int,
    tau_percentile: float = 80.0,
    chunk_boundary_mode: str = "first_only",
) -> EventBudget:
    """Per-sample event detection.

    For each batch item:
      1. Apply `chunk_boundary_mode` to derive forced event indices:
         - "all":        every chunk_size step in the valid window
         - "first_only": only the first valid step (lightweight anchor)
         - "none":       no forced events; rely entirely on saliency
      2. Pick steps where `s > τ` (τ = `tau_percentile` of NON-ZERO saliency).
      3. Apply NMS with `min_gap = nms_gap` to threshold events only.
      4. Combine forced ∪ threshold events; cap to `m_max` (forced protected).

    Returns padded `[B, M_max]` event indices with mask + count.
    """
    if chunk_boundary_mode not in ("all", "first_only", "none"):
        raise ValueError(
            f"chunk_boundary_mode must be 'all'/'first_only'/'none', got {chunk_boundary_mode!r}"
        )
    if saliency.ndim != 2:
        raise ValueError(f"saliency must be [B, T], got {tuple(saliency.shape)}")
    if saliency.shape != mask.shape:
        raise ValueError("saliency and mask must have the same shape")

    B, T = saliency.shape
    device = saliency.device

    chunk_size = max(1, int(chunk_size))
    nms_gap = max(0, int(nms_gap))

    saliency_cpu = saliency.detach().to("cpu", non_blocking=False)
    mask_cpu = mask.detach().to("cpu", non_blocking=False).bool()
    saliency_np = saliency_cpu.float().numpy()
    mask_np = mask_cpu.numpy()

    event_indices_cpu = [[] for _ in range(B)]
    for b in range(B):
        valid_steps = mask_np[b].nonzero()[0].tolist()
        if not valid_steps:
            continue
        valid_set = set(valid_steps)
        first_valid = valid_steps[0]
        last_valid = valid_steps[-1]

        if chunk_boundary_mode == "all":
            chunk_boundaries = [
                i
                for i in range(first_valid, last_valid + 1, chunk_size)
                if i in valid_set
            ]
        elif chunk_boundary_mode == "first_only":
            chunk_boundaries = [first_valid]
        else:
            chunk_boundaries = []

        sal_row = saliency_np[b]
        valid_sal = sal_row[valid_steps]
        nz = valid_sal[valid_sal > 1e-6]
        if nz.size > 0:

            import numpy as np

            tau = float(np.quantile(nz, tau_percentile / 100.0))
        else:
            tau = 0.0

        raw_threshold = [int(i) for i in valid_steps if float(sal_row[i]) > tau]

        if nms_gap > 0 and raw_threshold:
            ranked = sorted(raw_threshold, key=lambda i: -float(sal_row[i]))
            kept: list[int] = []
            for i in ranked:
                if all(abs(i - k) >= nms_gap for k in kept):
                    kept.append(i)
            threshold_events = sorted(kept)
        else:
            threshold_events = sorted(raw_threshold)

        merged = sorted(set(threshold_events) | set(chunk_boundaries))
        if len(merged) > m_max:
            forced = set(chunk_boundaries)
            scored = sorted(
                (i for i in merged if i not in forced),
                key=lambda i: -float(sal_row[i]),
            )
            keep_extra = m_max - len(forced)
            kept_set = forced | set(scored[: max(0, keep_extra)])
            merged = sorted(kept_set)
            if len(merged) > m_max:
                merged = merged[-m_max:]

        event_indices_cpu[b] = merged

    event_indices = torch.full((B, m_max), -1, dtype=torch.long, device=device)
    event_mask = torch.zeros((B, m_max), dtype=torch.bool, device=device)
    num_events = torch.zeros(B, dtype=torch.long, device=device)
    flat_indices = []
    flat_b = []
    flat_k = []
    for b, evs in enumerate(event_indices_cpu):
        n = len(evs)
        for k, idx in enumerate(evs):
            flat_b.append(b)
            flat_k.append(k)
            flat_indices.append(idx)
        num_events[b] = n
    if flat_indices:
        bt = torch.tensor(flat_b, dtype=torch.long, device=device)
        kt = torch.tensor(flat_k, dtype=torch.long, device=device)
        it = torch.tensor(flat_indices, dtype=torch.long, device=device)
        event_indices[bt, kt] = it
        event_mask[bt, kt] = True

    return EventBudget(
        event_indices=event_indices,
        event_mask=event_mask,
        num_events=num_events,
        saliency=saliency,
        n_action=torch.zeros_like(saliency),
        n_visual=torch.zeros_like(saliency),
    )


def detect_events_single(
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    chunk_size: int,
    m_max: int,
    nms_gap: int,
    tau_percentile: float = 80.0,
    chunk_boundary_mode: str = "first_only",
) -> dict:
    """Action-only event detection for a single sample.

    Returns dict with:
      event_indices: list[int]  (length ≤ m_max)
      event_mask:    list[bool] (length = m_max, True for valid events)
      seg_start:     list[int]  (length = m_max, step index of segment start; 0 for pad)
      seg_end:       list[int]  (length = m_max, step index of segment end; 0 for pad)
    """
    if chunk_boundary_mode not in ("all", "first_only", "none"):
        raise ValueError(
            f"chunk_boundary_mode must be 'all'/'first_only'/'none', got {chunk_boundary_mode!r}"
        )
    if actions.ndim != 2:
        raise ValueError(f"actions must be [N, A], got {actions.shape}")
    if mask.shape != (actions.shape[0],):
        raise ValueError(f"mask shape mismatch: {mask.shape} vs N={actions.shape[0]}")
    N = actions.shape[0]
    chunk_size = max(1, int(chunk_size))
    nms_gap = max(0, int(nms_gap))

    n_action_raw = np.zeros(N, dtype=np.float32)
    if N >= 2:
        diffs = np.linalg.norm(actions[1:] - actions[:-1], axis=-1)
        valid_pair = mask[1:].astype(bool) & mask[:-1].astype(bool)
        n_action_raw[1:] = diffs * valid_pair.astype(np.float32)

    nz = n_action_raw[n_action_raw > 1e-6]
    sigma_a = float(np.median(nz)) + 1e-8 if nz.size > 0 else 1.0
    saliency = n_action_raw / sigma_a

    valid_steps = np.nonzero(mask)[0].tolist()
    if not valid_steps:
        return {
            "event_indices": [],
            "event_mask": [False] * m_max,
            "seg_start": [0] * m_max,
            "seg_end": [0] * m_max,
        }
    valid_set = set(valid_steps)
    first_valid = valid_steps[0]
    last_valid = valid_steps[-1]

    if chunk_boundary_mode == "all":
        chunk_boundaries = [
            i for i in range(first_valid, last_valid + 1, chunk_size) if i in valid_set
        ]
    elif chunk_boundary_mode == "first_only":
        chunk_boundaries = [first_valid]
    else:
        chunk_boundaries = []

    valid_sal = saliency[valid_steps]
    nz_sal = valid_sal[valid_sal > 1e-6]
    tau = float(np.quantile(nz_sal, tau_percentile / 100.0)) if nz_sal.size > 0 else 0.0

    raw_threshold = [int(i) for i in valid_steps if float(saliency[i]) > tau]
    if nms_gap > 0 and raw_threshold:
        ranked = sorted(raw_threshold, key=lambda i: -float(saliency[i]))
        kept: list[int] = []
        for i in ranked:
            if all(abs(i - k) >= nms_gap for k in kept):
                kept.append(i)
        threshold_events = sorted(kept)
    else:
        threshold_events = sorted(raw_threshold)

    merged = sorted(set(threshold_events) | set(chunk_boundaries))
    if len(merged) > m_max:
        forced = set(chunk_boundaries)
        scored = sorted(
            (i for i in merged if i not in forced),
            key=lambda i: -float(saliency[i]),
        )
        keep_extra = m_max - len(forced)
        kept_set = forced | set(scored[: max(0, keep_extra)])
        merged = sorted(kept_set)
        if len(merged) > m_max:
            merged = merged[-m_max:]

    event_indices_pad = [-1] * m_max
    event_mask_list = [False] * m_max
    seg_start_list = [0] * m_max
    seg_end_list = [0] * m_max
    prev = first_valid
    for k, idx in enumerate(merged):
        event_indices_pad[k] = idx
        event_mask_list[k] = True
        seg_start_list[k] = prev
        seg_end_list[k] = idx
        prev = idx

    return {
        "event_indices": event_indices_pad,
        "event_mask": event_mask_list,
        "seg_start": seg_start_list,
        "seg_end": seg_end_list,
    }


def detect_events_causal_single(
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    m_max: int,
    tau_action: float = 0.30,
    max_seg_steps: int = 40,
    min_gap: int = 4,
    live_tail: bool = False,
) -> dict:
    """Append-only event detection for a single sample.

    Returns the same dict layout as `detect_events_single` so the two are
    drop-in interchangeable:
      event_indices: list[int]  (length m_max, -1 for pad)
      event_mask:    list[bool] (length m_max)
      seg_start:     list[int]  (length m_max, 0 for pad)
      seg_end:       list[int]  (length m_max, 0 for pad)

    Args:
        actions: `[N, A]` normalized actions.
        mask: `[N]` validity mask (False for front-padded slots).
        m_max: keep at most this many events, most recent first (ring buffer).
        tau_action: absolute threshold on `||a_i - a_{i-1}||`.
        max_seg_steps: force an event when a segment reaches this many steps.
        min_gap: refractory period, in steps, after a committed event.
        live_tail: also commit the last valid step, so the most recent motion
            always belongs to a segment. Closes the tail blind spot at the cost
            of one always-fresh (uncacheable) segment.
    """
    if actions.ndim != 2:
        raise ValueError(f"actions must be [N, A], got {actions.shape}")
    if mask.shape != (actions.shape[0],):
        raise ValueError(f"mask shape mismatch: {mask.shape} vs N={actions.shape[0]}")
    if m_max <= 0:
        raise ValueError(f"m_max must be positive, got {m_max}")

    max_seg_steps = max(1, int(max_seg_steps))
    min_gap = max(1, int(min_gap))
    tau_action = float(tau_action)

    empty = {
        "event_indices": [-1] * m_max,
        "event_mask": [False] * m_max,
        "seg_start": [0] * m_max,
        "seg_end": [0] * m_max,
    }

    valid_steps = np.nonzero(mask)[0]
    if valid_steps.size == 0:
        return empty
    valid_set = set(int(i) for i in valid_steps.tolist())
    first_valid = int(valid_steps[0])
    last_valid = int(valid_steps[-1])

    committed = [first_valid]
    for i in range(first_valid + 1, last_valid + 1):
        if i not in valid_set:
            continue
        if i - committed[-1] < min_gap:
            continue

        if (i - 1) in valid_set:
            novelty = float(np.linalg.norm(actions[i] - actions[i - 1]))
        else:
            novelty = 0.0
        if novelty > tau_action or (i - committed[-1]) >= max_seg_steps:
            committed.append(i)

    if live_tail and committed[-1] != last_valid:
        committed.append(last_valid)

    segments = [
        (committed[k - 1] if k > 0 else committed[k], committed[k])
        for k in range(len(committed))
    ]
    segments = segments[-m_max:]

    event_indices_pad = [-1] * m_max
    event_mask_list = [False] * m_max
    seg_start_list = [0] * m_max
    seg_end_list = [0] * m_max
    for k, (seg_s, seg_e) in enumerate(segments):
        event_indices_pad[k] = seg_e
        event_mask_list[k] = True
        seg_start_list[k] = seg_s
        seg_end_list[k] = seg_e

    return {
        "event_indices": event_indices_pad,
        "event_mask": event_mask_list,
        "seg_start": seg_start_list,
        "seg_end": seg_end_list,
    }
