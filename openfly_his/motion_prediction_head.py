"""Weight-compatible token/state head for offline probes and deployment."""
import torch
from torch import nn
from torch.nn import functional as F


class MotionPredictionHead(nn.Module):
    def __init__(self, hidden_size, state_dim=4, history_steps=12, action_dim=4,
                 history_width=128, widths=(512, 256), arm="fusion", history_dropout=0.0):
        super().__init__()
        if arm not in ("token", "history", "fusion", "state") or not 0 <= history_dropout < 1:
            raise ValueError("Unknown arm or invalid history dropout")
        self.arm, self.history_steps, self.action_dim = arm, history_steps, action_dim
        self.history_dropout = history_dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_steps*(action_dim+1), history_width), nn.SiLU(),
            nn.Linear(history_width, history_width), nn.SiLU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(hidden_size+history_width+64, widths[0]), nn.SiLU(),
                                   nn.Linear(widths[0], widths[1]), nn.SiLU())
        self.classifier = nn.Linear(widths[1], 4)
        self.regressor = nn.Linear(widths[1], 4*2)

    def forward(self, hidden, states, history, history_mask):
        if history.shape[1:] != (self.history_steps, self.action_dim):
            raise ValueError("History shape differs from head configuration")
        if history_mask.shape != history.shape[:2]:
            raise ValueError("History mask shape differs from actions")
        mask = history_mask.bool()
        safe_history = torch.where(mask[..., None], history, torch.zeros_like(history))
        if self.training and self.history_dropout:
            keep = torch.rand(len(history), 1, device=history.device) >= self.history_dropout
            mask = mask & keep
            safe_history = torch.where(mask[..., None], safe_history, torch.zeros_like(safe_history))
        encoded = self.history_encoder(torch.cat((safe_history, mask[..., None].to(history.dtype)), -1).flatten(1))
        token = self.norm(hidden) if self.arm in ("token", "fusion") else torch.zeros_like(hidden)
        if self.arm not in ("history", "fusion"):
            encoded = torch.zeros_like(encoded)
        z = self.trunk(torch.cat((token, encoded, self.state_encoder(states)), -1))
        coefficients = self.regressor(z).reshape(-1, 4, 2)
        coefficients = torch.stack((F.softplus(coefficients[..., 0]), coefficients[..., 1]), -1)
        return self.classifier(z), coefficients
