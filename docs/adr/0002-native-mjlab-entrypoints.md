# 0002: Use stock mjlab entry points and manager lifecycle

Status: accepted; supersedes the custom environment and CLI placement in ADR 0001.

## Context

The first structural refactor separated task configuration and tensor MDP heads but
retained a custom environment loop and command-line launcher in `rl/runner.py`.
This duplicated mjlab's launch lifecycle and did not provide HDMI's concise
`train TASK` / `play TASK` workflow.

## Decision

Register `Mjlab-SuperMimic-Tracking-Unitree-G1` through the `mjlab.tasks` package
entry point using native `ManagerBasedRlEnvCfg` and a custom `SuperMimicRunner`.
Use mjlab's unmodified `train` and `play` executables. The runner implements
`learn`, `save`, `load`, `get_inference_policy` and logging integration. Quantitative
all-clip evaluation remains a separate `scripts/evaluate.py` entry point.

Replace the custom simulation loop with native command/action/observation/reward/
termination/event terms. The motion command owns reference progress, task runtime,
physical context and curriculum state. Keep analytic tensor heads separate so the
same equations remain differentiable in imagined rollouts.

Use a narrow replay adapter around the environment created by mjlab. Training uses
manual reset to capture terminal states; play uses native auto-reset and its supplied
observation dictionary. Evaluation masks scoring after each clip's first completion.
The adapter rejects entities absent from the learned task state.

Native reward and termination run before command advancement. They compare fresh
integrated coordinates to reference `f + 1`. Step/interval events follow reward in
mjlab's lifecycle, and their changes appear in the next observation/replay state.

## Consequences

Launcher/device/log/viewer behavior follows mjlab rather than a second project CLI.
The learned world model, BPTT policy gradient and critic remain SuperMimic; adopting
mjlab's runner interface does not select PPO. Native scene composition supports
future object tasks, which must still provide an explicit dynamics/MDP contract.

Resuming refills replay because checkpoints do not contain replay transitions.
Policy-only loading avoids the training allocation cost during play/evaluation.
Current training is single-GPU; multiple GPU requests must fail explicitly until a
synchronized distributed optimizer is implemented.
