"""SuperMimic learner configuration consumed by mjlab's standard CLI."""

from dataclasses import dataclass, field
from typing import Literal

import tyro
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    """Paper Table A.5 defaults; scene and observation settings belong to the task."""

    class_name: str = "SuperMimicRunner"
    seed: int = 0
    max_iterations: int = 320_000
    save_interval: int = 2000
    experiment_name: str = "supermimic_g1_tracking"
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "supermimic"
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("policy_history", "goal")}
    )
    # The inherited PPO rollout length is unused; collection uses utd.
    num_steps_per_env: tyro.conf.Suppress[int] = 0

    buffer_steps: int = 1000
    """Replay capacity in vectorized simulator steps."""
    utd: float = 0.4
    """Learner updates per vectorized simulator step."""
    horizon: int = 32
    """World-model imagination length for policy and critic training."""
    wm_window: int = 8
    """Autoregressive world-model training window length."""
    wm_batches: int = 10
    """World-model optimizer updates per learning iteration."""
    wm_batch: int = 2048
    """Real replay windows in each world-model optimizer update."""
    pi_batch: int = 4096
    """Imagined trajectories per policy update."""
    lr: float = 2e-4
    """Initial learning rate for all three optimizers."""
    gamma: float = 0.99
    """Reward discount used by the learner and reset curriculum."""
    lam: float = 0.95
    """Lambda-return coefficient for imagined critic targets."""
    action_noise: float = 0.1
    """Standard deviation of action noise during real replay collection."""
    critic_epochs: int = 5
    """Critic epochs per policy update."""
    critic_minibatches: int = 4
    """Critic minibatches per epoch."""
    no_critic: bool = False
    """Disable critic fitting and terminal value bootstrapping."""
    warmup_steps: int = 64
    """Replay prefill before updates, including after resuming."""
    wm_only_iters: int = 500
    """Initial learning iterations that update only the world model."""
    lr_decay_iters: int = 250_000
    """Cosine learning-rate decay horizon, ending at 3% of the initial rate."""
    log_every: int = 50
    """Learning iterations between metric reports."""
    no_compile: bool = False
    """Disable torch.compile for debugging and short smoke runs."""
    amp: Literal["none", "bf16"] = "bf16"
    """CUDA MLP autocast precision; state and reward calculations remain FP32."""
