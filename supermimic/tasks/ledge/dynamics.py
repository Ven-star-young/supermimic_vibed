"""Robot dynamics conditioned on position and heading in the fixed ledge scene."""

import torch

from ...mdp.state import G1Dynamics


def scene_heading(S):
    forward = S[1][..., :2, 0]
    xy = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.stack((xy[..., 1], xy[..., 0]), -1)


class LedgeDynamics(G1Dynamics):
    feature_dim = G1Dynamics.feature_dim + 4
    input_dim = G1Dynamics.input_dim + 4

    @staticmethod
    def features(S):
        # G1 invariant features already include height; xy and heading recover
        # the fixed wall/edge location that the flat-ground model discards.
        return torch.cat((G1Dynamics.features(S), S[0][..., :2], scene_heading(S)), -1)
