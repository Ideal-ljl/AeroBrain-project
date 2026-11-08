"""Frozen predicted-class/L/Phi templates as a flow-matching source.

Only observations enter this module. Adjust physical templates first, then
apply each environment's MIN_MAX mapping. Stop noise stays N(0, 1).
"""
import math
import torch
from torch import nn
from .motion_prediction_head import MotionPredictionHead
from .motion_prior_config import resolve_motion_prior_config


class TemplateMotionPrior(nn.Module):
    uses_vlm_features = True
    template_conditioned = True
    fixed_components = True
    full_fixed = False

    def __init__(self, hidden_size, spec):
        super().__init__()
        self.spec = resolve_motion_prior_config(spec, horizon=spec['horizon'], state_dim=spec['state_dim'])
        self.horizon, self.K = self.spec['horizon'], 4
        settings = dict(self.spec['head_settings'])
        if settings['hidden_size'] != hidden_size:
            raise ValueError('Imported motion head and VLM hidden sizes differ')
        self.prediction_head = MotionPredictionHead(**settings)
        self.fixed_buffer_keys = ('template_distance', 'template_yaw', 'template_z',
                                  'action_min', 'action_max', 'length_scale', 'noise_cholesky', 'logit_shift')
        for key in self.fixed_buffer_keys:
            self.register_buffer(key, torch.tensor(self.spec[key], dtype=torch.float32))

    @torch.no_grad()
    def reset_basis(self):
        for key in self.fixed_buffer_keys:
            buffer = getattr(self, key)
            buffer.copy_(torch.tensor(self.spec[key], device=buffer.device, dtype=buffer.dtype))

    def environment_ids(self, names, *, device=None):
        lookup = {name: i for i, name in enumerate(self.spec['environments'])}
        if isinstance(names, str):
            names = [names]
        try:
            return torch.tensor([lookup[name] for name in names], device=device, dtype=torch.long)
        except KeyError as error:
            raise ValueError(f'Unknown template-prior environment: {error.args[0]}') from error

    def forward(self, history_hidden, states, *, environment=None):
        if environment is None:
            raise ValueError('Template prior requires explicit motion_environment; never infer it from state')
        if states.ndim == 3 and states.shape[1] == 1:
            states = states[:, 0]
        batch = len(states)
        if states.shape != (batch, self.spec['state_dim']) or not torch.isfinite(states).all():
            raise ValueError('Invalid template-prior states')
        if (history_hidden is None or history_hidden.shape != (batch, self.spec['head_settings']['hidden_size'])
                or not torch.isfinite(history_hidden).all()):
            raise ValueError('Invalid template-prior history token')
        environment = torch.as_tensor(environment, device=states.device)
        if (environment.dtype != torch.long or environment.shape != (batch,)
                or (environment < 0).any() or (environment >= len(self.length_scale)).any()):
            raise ValueError('Invalid motion_environment IDs')
        with torch.autocast(device_type=states.device.type, enabled=False):
            history = states.new_zeros(batch, self.prediction_head.history_steps, 4, dtype=torch.float32)
            mask = torch.zeros(history.shape[:2], device=states.device, dtype=torch.bool)
            logits, coefficients = self.prediction_head(history_hidden.float(), states.float(), history, mask)
        if not torch.isfinite(logits).all() or not torch.isfinite(coefficients).all():
            raise FloatingPointError('Nonfinite template-prior prediction')
        return logits + self.logit_shift, coefficients, environment

    def adjusted_mean(self, parameters):
        logits, coefficients, environment = parameters
        mode = logits.argmax(-1)
        row = torch.arange(len(logits), device=logits.device)
        selected = coefficients[row, mode]
        length = selected[:, 0].clamp(0, self.spec['max_normalized_length']) * self.length_scale[environment]
        phi = (selected[:, 1] * math.pi).clamp(-self.spec['max_abs_yaw'], self.spec['max_abs_yaw'])
        yaw = self.template_yaw[mode]
        # Additive correction avoids dividing by a straight template's near-zero yaw.
        yaw = yaw + (phi - yaw.sum(-1))[:, None] / self.horizon
        yaw = yaw.clamp(-math.pi/2 + 1e-4, math.pi/2 - 1e-4)
        distance = self.template_distance[mode] * length[:, None]
        physical = torch.stack((distance*yaw.cos(), distance*yaw.sin(),
                                self.template_z[mode]*length[:, None]), -1)
        lo, hi = self.action_min[environment, None], self.action_max[environment, None]
        mean = (physical-lo)/(hi-lo+1e-8)*2-1
        # LeRobot's legacy MIN_MAX maps a constant coordinate to -1, not 0.
        # Ignore the vertical prototype on planar GS environments.
        mean = torch.where(hi > lo, mean, torch.full_like(mean, -1.))
        return mean, mode

    @torch.no_grad()
    def sample(self, history_hidden, states=None, *, generator=None, parameters=None, environment=None):
        parameters = self(history_hidden, states, environment=environment) if parameters is None else parameters
        with torch.autocast(device_type=parameters[0].device.type, enabled=False):
            mean, mode = self.adjusted_mean(parameters)
            eps = torch.randn(len(mode), self.horizon, 4, device=mean.device, generator=generator)
            residual = torch.bmm(self.noise_cholesky[mode].float(),
                                 eps[..., :3].reshape(len(mode), self.horizon*3, 1)).reshape_as(mean)
            movable = self.action_max[parameters[2], None] > self.action_min[parameters[2], None]
            xyz = torch.where(movable, mean + self.spec['noise_scale']*residual, mean)
            result = torch.cat((xyz, eps[..., 3:]), -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Nonfinite template-prior source')
        return result
