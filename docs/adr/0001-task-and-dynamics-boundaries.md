# 0001: Separate task composition from simulation and differentiable learning

Status: accepted; environment lifecycle and CLI placement superseded by [ADR 0002](0002-native-mjlab-entrypoints.md)

## Context

The first mjlab migration concentrated robot setup, scene construction, resets and physical
context in one environment module. The MDP module mixed G1-specific state layout with all
analytic heads, and training/evaluation initialized a run at import time. This made adding
an interacting object likely to require unrelated changes across simulation and learning,
with hard-coded dimensions capable of silently excluding object state.

HDMI's assets, task configuration and MDP term separation is a useful reference. SuperMimic
also needs an explicit differentiable dynamics contract: its actor is optimized through a
learned world model, rather than HDMI's PPO transitions.

## Decision

Use `assets/`, `tasks/`, `envs/`, `mdp/` and `rl/` packages. The current G1 task assembles
native mjlab scene/simulation settings with command, action, event, observation, reward and
termination configuration. Keep analytic heads as tensor functions available to both real
and imagined rollouts. Keep native simulator state/reset handling in the environment.

Task construction supplies dimensions and checkpoint metadata. Models and replay receive
task dimensions rather than depending on training-script constants. Task state encoding,
world-model features, regression targets, integration and losses form a single contract.
Training and evaluation entry points are import-safe and live in `rl.runner` and
`rl.evaluation`. Top-level compatibility adapters were initially retained during the
refactor, then removed following the explicit request to clean up old entry points.
Callers now use the package boundaries directly.

Permit native object entities and contact sensors in low-level simulation configurations.
Reject such additions through the current robot-only training factory. An object interaction
task must explicitly implement its state/dynamics, references, observation, reward and reset
contract before it can train.

## Consequences

G1 tracking remains the baseline and retains its existing algorithm and default physics.
Scene composition can be exercised without altering the optimizer, while future task code
has a clear place for object state and contact objectives. Checkpoint metadata detects
incompatible layouts earlier.

The reference command also makes one small curriculum correction: when several worlds
finish from the same bin in one step, their returns are averaged before the bin's EMA
update. The old duplicate indexed writes had undefined winner semantics on GPU. Sampling
mixture weights and EMA rates are unchanged.

This does not make every existing G1 formula robot-agnostic, nor does it implement manipulation.
Task implementations still need domain-specific motion data, state parameterization and
differentiable contact modeling. Tests must verify both physical simulation behavior and
the gradient path from reward through the learned dynamics to actions.
