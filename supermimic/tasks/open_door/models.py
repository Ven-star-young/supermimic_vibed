"""Use the shared tracking networks with OpenDoor's tensor dimensions."""

import torch

from ..g1_tracking.models import Critic as TrackingCritic
from ..g1_tracking.models import Policy as TrackingPolicy
from ..g1_tracking.models import WorldModel as TrackingWorldModel
from .dynamics import DoorDynamics


class Policy(TrackingPolicy):
    """102-D history and 75-D future references, with no geometry encoder."""

    def __init__(self, history_length=10, goal_frames=10):
        super().__init__(obs_dim=102, goal_dim=75 * goal_frames, hist=history_length)


class WorldModel(TrackingWorldModel):
    """80 state features + 29 actions + 43 context values -> 72 increments."""

    def __init__(self, dynamics=None, hidden=(2048, 2048, 2048)):
        super().__init__(hidden=hidden, dynamics=DoorDynamics() if dynamics is None else dynamics)

    def inputs(self, S, action, normalized_context):
        # Hinge xyz already enters through relative features, so avoid adding
        # absolute coordinates as a second input to the dynamics MLP.
        return torch.cat((self.dynamics.features(S, normalized_context), action,
                          normalized_context[..., :33], normalized_context[..., 36:46]), -1)


class Critic(TrackingCritic):
    """205 current privileged features plus 75 channels per future frame."""

    def __init__(self, goal_frames=10):
        self.in_dim = 205 + 75 * goal_frames
        super().__init__(self.in_dim)
