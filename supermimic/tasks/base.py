"""Task-owned dimensions and model compatibility, independent of the learner."""
from dataclasses import asdict, dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskDimensions:
    state: int
    action: int
    context: int
    observation: int
    goal_per_frame: int
    critic_features: int


@dataclass
class TrackingTask:
    name: str
    motion: object
    mdp: object
    dynamics: object
    env_cfg: object
    modeled_entities: tuple[str, ...] = ("robot",)
    play: bool = False
    validate_env_cfg: Callable | None = None

    @property
    def dimensions(self):
        d = self.dynamics
        return TaskDimensions(d.state_dim, d.action_dim, d.context_dim, self.mdp.observation_dim,
                              self.motion.goal_feature_dim, self.mdp.critic_feature_dim)

    @property
    def max_step_reward(self):
        return self.mdp.max_step_reward

    def goal_dim(self, frames):
        return self.dimensions.goal_per_frame * frames

    def make_env(self, num_envs, device="cuda:0", gamma=0.99, **kwargs):
        from ..envs.tracking_env import TrackingSim
        if self.validate_env_cfg is not None:
            self.validate_env_cfg(self.env_cfg)
        extra = set(self.env_cfg.scene.entities) - set(self.modeled_entities)
        if extra:
            raise ValueError(
                f"Task {self.name} dynamics/state do not model scene entities {sorted(extra)}. "
                "Add object reference/state, dynamics and differentiable MDP heads before training. "
                "Use TrackingSim directly for scene/contact inspection."
            )
        kwargs.setdefault("eval_mode", self.play)
        return TrackingSim(num_envs, self.motion, self.mdp, device=device, gamma=gamma,
                           cfg=self.env_cfg, dynamics=self.dynamics, **kwargs)

    def checkpoint_metadata(self):
        return {"name": self.name, "schema_version": 1, "dimensions": asdict(self.dimensions)}

    def validate_checkpoint(self, checkpoint):
        metadata = checkpoint.get("task")
        if metadata is None:
            legacy = TaskDimensions(71, 29, 33, 99, 73, 184)
            if self.name != "g1_tracking" or self.dimensions != legacy:
                raise ValueError("Legacy checkpoint has no task schema; only default G1 tracking is supported")
        elif metadata != self.checkpoint_metadata():
            raise ValueError(f"Checkpoint task/schema mismatch: {metadata} != {self.checkpoint_metadata()}")
