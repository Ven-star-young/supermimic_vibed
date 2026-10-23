"""Hard simulator failures and their differentiable imagination counterpart."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TerminationCfg:
    height: float = 0.25
    tilt: float = 0.8
    height_temperature: float = 0.02
    tilt_temperature: float = 0.04


def tracking_errors(S, reference_p, reference_R):
    p, R = S[0], S[1]
    return (reference_p[..., 2] - p[..., 2]).abs(), (reference_R[..., 2, 2] - R[..., 2, 2]).abs()


def tracking_termination(S, reference_p, reference_R, cfg: TerminationCfg):
    e_z, e_tilt = tracking_errors(S, reference_p, reference_R)
    return (e_z > cfg.height) | (e_tilt > cfg.tilt)


def soft_tracking_termination(S, reference_p, reference_R, cfg: TerminationCfg,
                              tz: float | None = None, tt: float | None = None):
    e_z, e_tilt = tracking_errors(S, reference_p, reference_R)
    tz = cfg.height_temperature if tz is None else tz
    tt = cfg.tilt_temperature if tt is None else tt
    alive = (1 - torch.sigmoid((e_z - cfg.height) / tz)) * (1 - torch.sigmoid((e_tilt - cfg.tilt) / tt))
    return 1 - alive
