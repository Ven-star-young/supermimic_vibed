"""Task-owned tensor spaces and checkpoint compatibility.

Environment construction belongs to mjlab's task registry and standard scripts.
This runtime only describes the differentiable task used by the learner.
"""
from dataclasses import asdict, dataclass


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

    def validate_model_coverage(self):
        extra = set(self.env_cfg.scene.entities) - set(self.modeled_entities)
        if extra:
            raise ValueError(
                f"Task {self.name} dynamics/state do not model scene entities {sorted(extra)}. "
                "Extend object references, state, dynamics and differentiable heads before training."
            )
        # Physical and imagined returns must be the same function. Customize the
        # shared command.reward_cfg, not a second simulator-only multiplier.
        from ..mdp.native import tracking_reward
        terms = self.env_cfg.rewards
        if (set(terms) != {"tracking"} or terms["tracking"].func is not tracking_reward
                or terms["tracking"].weight != 1.0 or self.env_cfg.scale_rewards_by_dt):
            raise ValueError("Simulator rewards differ from the differentiable task head; configure motion.reward_cfg")

    def checkpoint_metadata(self):
        return {"name": self.name, "schema_version": 2, "dimensions": asdict(self.dimensions)}

    def validate_checkpoint(self, checkpoint):
        metadata = checkpoint.get("task")
        if metadata is None:
            raise ValueError(
                "Legacy checkpoint has no task schema and may use the removed reference-orientation "
                "observation. Retrain with the current task; implicit weight conversion is unsupported."
            )
        elif metadata != self.checkpoint_metadata():
            raise ValueError(
                f"Checkpoint task/schema mismatch: {metadata} != {self.checkpoint_metadata()}. "
                "Retrain with the current observation and reward definition."
            )
