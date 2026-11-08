"""Motion-adaptive memory encoders and event selection."""

from airbrain.memory.mam.encoder import MAMMemoryEncoder
from airbrain.memory.mam.saliency import (
    EventBudget,
    compute_action_novelty,
    compute_saliency,
    compute_visual_novelty,
    detect_events,
    detect_events_causal_single,
    detect_events_single,
)
from airbrain.memory.mam.segment_encoder import (
    ActionSegmentEncoder,
    SegmentCrossAttention,
)

__all__ = [
    "ActionSegmentEncoder",
    "EventBudget",
    "MAMMemoryEncoder",
    "SegmentCrossAttention",
    "compute_action_novelty",
    "compute_saliency",
    "compute_visual_novelty",
    "detect_events",
    "detect_events_causal_single",
    "detect_events_single",
]
