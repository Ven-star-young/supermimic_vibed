"""Robot and entity references on one clock; targets stay outside dynamics."""

import torch

from ...rotations import mat_to_6d, quat_to_mat, yaw_mat
from ..interaction.motion import InteractionMotionLib


class EntityMotionLib(InteractionMotionLib):
    object_goal_feature_dim = 9

    def __init__(self, files, kin, device, *, object_enabled=True, max_objects=8, extra_objects=()):
        super().__init__(files, kin, device, object_enabled=object_enabled)
        if max_objects < 1 or int(object_enabled) + len(extra_objects) > max_objects:
            raise ValueError("Entity references exceed max_objects")
        self.max_objects = max_objects
        shape = (self.num_frames, max_objects, 3)
        position, rotation, velocity, angular_velocity = self.object_p, self.object_R, self.object_v, self.object_w
        self.object_p = position.new_zeros(shape)
        self.object_R = torch.eye(3, device=device).expand(*shape, 3).clone()
        self.object_v = position.new_zeros(shape)
        self.object_w = position.new_zeros(shape)
        self.valid_mask = position.new_zeros(max_objects)
        self.dynamic_mask = position.new_zeros(max_objects)
        self.target_mask = position.new_zeros(max_objects)
        if object_enabled:
            self.object_p[:, 0], self.object_R[:, 0] = position, rotation
            self.object_v[:, 0], self.object_w[:, 0] = velocity, angular_velocity
            self.valid_mask[0] = self.dynamic_mask[0] = self.target_mask[0] = 1.
        for index, obj in enumerate(extra_objects, start=int(object_enabled)):
            self.object_p[:, index] = torch.tensor(obj.pos, device=device)
            self.object_R[:, index] = quat_to_mat(torch.tensor(obj.quat, device=device))
            self.valid_mask[index], self.dynamic_mask[index] = 1., float(obj.dynamic)

    def object_goal(self, S, f, K):
        """N x K x 9 future poses, anchored to the current actual G1 heading.

        Non-target objects have no task trajectory and are masked to zero.
        Future frames clamp within each clip, never crossing into another one.
        """
        future = self.future_frames(f, K)
        Ht = yaw_mat(S[1]).transpose(-1, -2)[..., None, None, :, :]
        position = self.object_p[future].movedim(-3, -2)
        rotation = self.object_R[future].movedim(-4, -3)
        relative = (Ht @ (position - S[0][..., None, None, :]).unsqueeze(-1)).squeeze(-1)
        orientation = mat_to_6d(Ht @ rotation)
        goal = torch.cat((relative, orientation), -1) * self.target_mask[:, None, None]
        return goal.flatten(-3)
