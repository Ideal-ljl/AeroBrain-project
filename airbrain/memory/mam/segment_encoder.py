"""MAM segment encoder.

Per detected event, a small cross-attention block produces `tokens_per_event`
memory tokens from:
  Q : action-segment content encoding for the segment that ends at this event
  K/V : patch features from the segment's start frame and end frame
        (gives the cross-attention access to a "differential" view across the segment)

Event age/recency is added after cross-attention by `MAMMemoryEncoder`, so the
cross-attended content token can be cached independently of runtime age.

The implementation runs the per-event cross-attention in a single batched call by
gathering features into a `[B·E, ...]` layout, where E = M_max.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionSegmentEncoder(nn.Module):
    """Turns a variable-length action segment into M query vectors.

    Strategy: 1-D convolutions across the time axis, followed by adaptive
    pooling to a fixed token count, then a Linear head into `d_model`.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        d_model: int,
        tokens_per_event: int,
        hidden_dim: int | None = None,
        max_age: int = 256,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.action_dim = int(action_dim)
        self.d_model = int(d_model)
        self.tokens_per_event = int(tokens_per_event)
        self.max_age = int(max_age)

        self.conv = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, d_model)
        # Kept here for checkpoint compatibility, but applied after visual
        # cross-attention so event content tokens can be cached by segment.
        self.age_embedding = nn.Embedding(max_age, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        # Per-token learned offsets that break the within-event symmetry. The
        # action-segment encoding produces ONE query per event; without these
        # offsets, expanding it to `tokens_per_event` queries would feed
        # identical Q rows to a position-wise cross-attention, so the output
        # tokens collapse to one repeated vector and `(tokens_per_event - 1)`
        # of every event's KV slots are wasted.
        self.token_offsets = nn.Parameter(torch.empty(tokens_per_event, d_model))
        nn.init.normal_(self.token_offsets, mean=0.0, std=0.02)

    def forward(
        self,
        segment_actions: torch.Tensor,         # [N, L_max, A]
        segment_lengths: torch.Tensor,         # [N] valid length per segment
        ages: torch.Tensor,                    # [N] retained for API compatibility
    ) -> torch.Tensor:
        """Return `[N, tokens_per_event, d_model]` query vectors."""
        if segment_actions.ndim != 3:
            raise ValueError(f"expected [N, L, A], got {tuple(segment_actions.shape)}")
        N, L_max, A = segment_actions.shape

        # Build mask [N, L_max] from lengths
        ar = torch.arange(L_max, device=segment_actions.device)[None, :]
        valid = ar < segment_lengths[:, None].clamp_min(0)
        x = segment_actions * valid.unsqueeze(-1).to(segment_actions.dtype)

        # [N, A, L_max]
        x = x.transpose(1, 2)
        h = self.conv(x)  # [N, hidden, L_max]

        # Adaptive avg pool over valid length: use masked mean instead of plain
        # adaptive pool to respect variable lengths.
        valid_t = valid.unsqueeze(1).to(h.dtype)  # [N, 1, L_max]
        h = h * valid_t
        valid_count = valid_t.sum(dim=-1).clamp_min(1.0)  # [N, 1]
        pooled = h.sum(dim=-1) / valid_count               # [N, hidden]

        q = self.head(pooled)                              # [N, d_model]
        q = self.query_norm(q)

        # Expand to `tokens_per_event` queries. The learned per-token offsets
        # break the within-event symmetry — without them every query row would
        # be identical and `nn.MultiheadAttention` (position-wise) would emit
        # `tokens_per_event` copies of the same vector.
        q = q.unsqueeze(1) + self.token_offsets.to(dtype=q.dtype).unsqueeze(0)
        return q  # [N, tokens_per_event, d_model]

    def age_tokens(self, ages: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return runtime age embeddings as `[N, 1, d_model]`.

        This is added after cross-attention, keeping the cross-attended event
        content independent of age/recency for inference-time caching.
        """
        age_clamped = ages.clamp(min=0, max=self.max_age - 1)
        emb = self.age_embedding(age_clamped).unsqueeze(1)
        return emb.to(dtype=dtype) if dtype is not None else emb


class _CrossAttnBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("num_heads must divide d_model")
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        queries: torch.Tensor,         # [N, M, d_model]
        keys_values: torch.Tensor,     # [N, K, d_model]
        kv_padding_mask: torch.Tensor | None = None,  # [N, K] True for PAD
    ) -> torch.Tensor:
        q = self.q_norm(queries)
        kv = self.kv_norm(keys_values)
        out, _ = self.attn(q, kv, kv, key_padding_mask=kv_padding_mask, need_weights=False)
        x = queries + out
        x = x + self.mlp(x)
        return x


class SegmentCrossAttention(nn.Module):
    """Cross-attention from action-query → segment patch features.

    Patch features come from a vision tower / projector and are assumed to
    live in `d_model` (VLM hidden) space. Caller is responsible for
    projecting if needed.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        patch_proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.d_model = int(d_model)
        self.patch_proj: nn.Module
        if patch_proj_dim is not None and patch_proj_dim != d_model:
            self.patch_proj = nn.Linear(int(patch_proj_dim), d_model)
        else:
            self.patch_proj = nn.Identity()
        self.layers = nn.ModuleList(
            [_CrossAttnBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,            # [N, M, d_model]
        kv_patches: torch.Tensor,         # [N, K, d_kv]
        kv_padding_mask: torch.Tensor | None = None,  # [N, K] True = PAD
    ) -> torch.Tensor:
        kv = self.patch_proj(kv_patches)
        x = queries
        for layer in self.layers:
            x = layer(x, kv, kv_padding_mask=kv_padding_mask)
        return self.out_norm(x)
