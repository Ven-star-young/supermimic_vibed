"""Task networks over raw, variable-size scenes and per-object reference tracks.

Only raw observations are retained in the actor history. Learned geometry and
scene embeddings are recomputed on every forward, so they cannot become stale
as the actor learns. Actor, critic and world model own independent encoders;
only the immutable geometry assets are shared between them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .entity_world_model import EntityGeometryBank, EntityGeometryEncoder
from .models import RunningNorm, mlp, run_mlp


def _masked_values(values, valid):
    return torch.where(valid[..., None], values, torch.zeros_like(values))


def _update_valid(norm, values, valid):
    """Pool statistics over valid entities and all batch/time dimensions."""
    selected = values.reshape(-1, values.shape[-1])[valid.reshape(-1)]
    if selected.numel():
        norm.update(selected)


def _geometry(bank, encoder, ids, valid):
    """IDs select assets only; their numerical values never enter a network."""
    ids = torch.where(valid, ids, torch.zeros_like(ids)).long()
    unique, inverse = ids.reshape(-1).unique(return_inverse=True)
    points, extents = bank.tensors(ids.device)
    encoded = encoder(points.index_select(0, unique), extents.index_select(0, unique))
    return _masked_values(encoded[inverse].reshape(*ids.shape, 64), valid)


def _pool(tokens, query, valid):
    """Permutation-invariant attention, with an exact zero for empty scenes."""
    scores = (tokens * query[..., None, :]).sum(-1) / math.sqrt(tokens.shape[-1])
    weights = scores.masked_fill(~valid, -1e4).softmax(-1) * valid.to(scores.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return (tokens * weights[..., None]).sum(-2)


class EntityPolicy(nn.Module):
    """G1 history + a masked set of observed objects + robot reference goals.

    Raw frame layout: robot 93, then N objects of
    ``pose9, valid, dynamic, target, geometry_id``. The actor sees no object
    velocities, masses or contact parameters. Target labels are task inputs.
    """

    robot_obs_dim, object_obs_dim = 93, 13

    def __init__(self, geometry_bank=None, max_objects=8, hist=10, goal_dim=730,
                 z_dim=32, action_dim=29, hidden=(2048, 1024, 512, 256), scene_dim=128):
        super().__init__()
        if max_objects < 1 or hist < 1:
            raise ValueError("max_objects and history must be positive")
        self.geometry_bank = geometry_bank if geometry_bank is not None else EntityGeometryBank()
        self.max_objects, self.hist, self.goal_dim = max_objects, hist, goal_dim
        self.obs_dim = self.robot_obs_dim + max_objects * self.object_obs_dim
        self.frame_dim = self.robot_obs_dim + scene_dim
        self.geometry_encoder = EntityGeometryEncoder()
        self.object_encoder = mlp(75, [128], scene_dim, act=nn.SiLU)
        self.scene_query = nn.Linear(self.robot_obs_dim, scene_dim)
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.dec = mlp(z_dim + self.frame_dim * hist, list(hidden), action_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.robot_norm = RunningNorm(self.robot_obs_dim)
        self.object_pose_norm = RunningNorm(9)
        self.goal_norm = RunningNorm(goal_dim)

    def split_observations(self, observations):
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected {self.obs_dim} raw observation values, got {observations.shape[-1]}")
        return (observations[..., :self.robot_obs_dim],
                observations[..., self.robot_obs_dim:].reshape(*observations.shape[:-1], self.max_objects, 13))

    def encode_scene(self, robot, objects):
        valid = objects[..., 9] > 0.5
        geometry = _geometry(self.geometry_bank, self.geometry_encoder, objects[..., 12], valid)
        pose = self.object_pose_norm(_masked_values(objects[..., :9], valid))
        features = _masked_values(torch.cat((pose, geometry, objects[..., 10:12]), -1), valid)
        tokens = _masked_values(run_mlp(self.object_encoder, features), valid)
        return _pool(tokens, run_mlp(self.scene_query, robot), valid)

    def forward(self, obs_hist, goal):
        if obs_hist.shape[-2] != self.hist:
            raise ValueError(f"Expected {self.hist} history frames")
        robot, objects = self.split_observations(obs_hist)
        robot = self.robot_norm(robot)
        scene = self.encode_scene(robot, objects)
        frames = torch.cat((robot, scene), -1).flatten(-2)
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        return run_mlp(self.dec, torch.cat((latent, frames), -1))

    @torch.no_grad()
    def update_normalizers(self, obs_hist, goal):
        robot, objects = self.split_observations(obs_hist)
        self.robot_norm.update(robot)
        _update_valid(self.object_pose_norm, objects[..., :9], objects[..., 9] > 0.5)
        self.goal_norm.update(goal)


class _CriticSceneLayer(nn.Module):
    """Friction-conditioned messages among robot, object and ground tokens."""

    def __init__(self, width):
        super().__init__()
        self.edge = mlp(3, [32], 32, act=nn.SiLU)
        self.message = mlp(width * 2 + 32, [width], width, act=nn.SiLU)
        self.update = mlp(width * 2, [width], width, act=nn.SiLU)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, friction, valid):
        n = tokens.shape[-2]
        receiver = tokens.unsqueeze(-2).expand(*tokens.shape[:-2], n, n, tokens.shape[-1])
        sender = tokens.unsqueeze(-3).expand_as(receiver)
        pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        pair_valid = pair_valid & ~torch.eye(n, dtype=torch.bool, device=tokens.device)
        edges = run_mlp(self.edge, _masked_values(friction, pair_valid))
        messages = run_mlp(self.message, torch.cat((receiver, sender, edges), -1))
        messages = _masked_values(messages, pair_valid).sum(-2)
        messages = messages / pair_valid.sum(-1, keepdim=True).clamp_min(1)
        result = self.norm(tokens + run_mlp(self.update, torch.cat((tokens, messages), -1)))
        return _masked_values(result, valid)


class EntityCritic(nn.Module):
    """Value conditioned on all relevant bodies and target-object futures.

    Flat input layout: robot184, N*object38, (N+2)^2*friction3,
    plane4, robot_goal730, N*K*future_pose9. Pair order is robot, objects,
    ground. Each object38 is ``state15, mass/com/inertia10, dynamic,
    reference_error9, target, valid, geometry_id``. Future poses use the same
    current robot-heading frame as current object poses, and are object-major.
    """

    robot_feature_dim, object_raw_dim = 184, 38

    def __init__(self, geometry_bank=None, max_objects=8, goal_frames=10, goal_dim=730,
                 hidden=(256, 128), scene_dim=128):
        super().__init__()
        if max_objects < 1 or goal_frames < 1:
            raise ValueError("max_objects and goal_frames must be positive")
        self.geometry_bank = geometry_bank if geometry_bank is not None else EntityGeometryBank()
        self.max_objects, self.goal_frames, self.goal_dim = max_objects, goal_frames, goal_dim
        self.num_entities = max_objects + 2
        self.current_dim = 184 + 38 * max_objects + 3 * self.num_entities**2 + 4
        self.in_dim = self.current_dim + goal_dim + max_objects * goal_frames * 9
        self.geometry_encoder = EntityGeometryEncoder()
        self.future_encoder = mlp(goal_frames * 9, [128, 64], 32, act=nn.SiLU)
        self.object_encoder = mlp(100 + 32, [128], scene_dim, act=nn.SiLU)
        self.robot_encoder = mlp(184, [128], scene_dim, act=nn.SiLU)
        self.ground_encoder = mlp(4, [64], scene_dim, act=nn.SiLU)
        self.scene_layers = nn.ModuleList([_CriticSceneLayer(scene_dim) for _ in range(2)])
        self.scene_query = nn.Linear(184, scene_dim)
        self.net = mlp(184 + scene_dim + goal_dim, list(hidden), 1)
        self.robot_norm = RunningNorm(184)
        self.object_norm = RunningNorm(25)
        self.error_norm = RunningNorm(9)
        self.future_norm = RunningNorm(9)
        self.friction_norm = RunningNorm(3)
        self.ground_norm = RunningNorm(4)
        self.goal_norm = RunningNorm(goal_dim)

    def split_inputs(self, values):
        if values.shape[-1] != self.in_dim:
            raise ValueError(f"Expected {self.in_dim} critic input values, got {values.shape[-1]}")
        shape = values.shape[:-1]
        offset = 184 + self.max_objects * 38
        pair_end = offset + 3 * self.num_entities**2
        robot = values[..., :184]
        objects = values[..., 184:offset].reshape(*shape, self.max_objects, 38)
        friction = values[..., offset:pair_end].reshape(*shape, self.num_entities, self.num_entities, 3)
        plane = values[..., pair_end:self.current_dim]
        goal = values[..., self.current_dim:self.current_dim + self.goal_dim]
        future = values[..., self.current_dim + self.goal_dim:].reshape(
            *shape, self.max_objects, self.goal_frames, 9)
        return robot, objects, friction, plane, goal, future

    def encode_scene(self, robot, objects, friction, plane, future):
        valid = objects[..., 36] > 0.5
        target = valid & (objects[..., 35] > 0.5)
        geometry = _geometry(self.geometry_bank, self.geometry_encoder, objects[..., 37], valid)
        physical = self.object_norm(_masked_values(objects[..., :25], valid))
        error = _masked_values(self.error_norm(_masked_values(objects[..., 26:35], target)), target)
        ref = self.future_norm(_masked_values(future, target[..., None].expand(*target.shape, self.goal_frames)))
        ref = _masked_values(run_mlp(self.future_encoder, ref.flatten(-2)), target)
        features = torch.cat((physical[..., :15], geometry, physical[..., 15:25],
                              objects[..., 25:26], error, target[..., None].to(objects.dtype)), -1)
        object_tokens = _masked_values(run_mlp(self.object_encoder, torch.cat((
            _masked_values(features, valid), ref), -1)), valid)
        tokens = torch.cat((run_mlp(self.robot_encoder, robot).unsqueeze(-2), object_tokens,
                            run_mlp(self.ground_encoder, self.ground_norm(plane)).unsqueeze(-2)), -2)
        one = torch.ones_like(valid[..., :1])
        entity_valid = torch.cat((one, valid, one), -1)
        pair_valid = entity_valid.unsqueeze(-1) & entity_valid.unsqueeze(-2)
        friction = _masked_values(self.friction_norm(_masked_values(friction, pair_valid)), pair_valid)
        for layer in self.scene_layers:
            tokens = layer(tokens, friction, entity_valid)
        # Ground participates even in an empty scene. The robot conditions the
        # query and messages; its original features also reach the final MLP.
        return _pool(tokens[..., 1:, :], run_mlp(self.scene_query, robot), entity_valid[..., 1:])

    def forward(self, values):
        robot, objects, friction, plane, goal, future = self.split_inputs(values)
        robot = self.robot_norm(robot)
        scene = self.encode_scene(robot, objects, friction, plane, future)
        return run_mlp(self.net, torch.cat((robot, scene, self.goal_norm(goal)), -1)).squeeze(-1)

    @torch.no_grad()
    def update_normalizers(self, values):
        robot, objects, friction, plane, goal, future = self.split_inputs(values)
        valid = objects[..., 36] > 0.5
        target = valid & (objects[..., 35] > 0.5)
        self.robot_norm.update(robot)
        _update_valid(self.object_norm, objects[..., :25], valid)
        _update_valid(self.error_norm, objects[..., 26:35], target)
        _update_valid(self.future_norm, future, target[..., None].expand(*target.shape, self.goal_frames))
        one = torch.ones_like(valid[..., :1])
        entities = torch.cat((one, valid, one), -1)
        pair_valid = entities.unsqueeze(-1) & entities.unsqueeze(-2)
        pair_valid = pair_valid & ~torch.eye(self.num_entities, dtype=torch.bool, device=values.device)
        _update_valid(self.friction_norm, friction, pair_valid)
        self.ground_norm.update(plane)
        self.goal_norm.update(goal)
