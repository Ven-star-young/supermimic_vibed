# 0003: Align the actor observation with the paper

Status: accepted.

## Context

The original independent implementation added the six-dimensional rotation
`rot6(R_actual.T @ R_reference)` to the policy observation. The simulator and native
entry-point migrations retained it to preserve behavior. The user now requests the
paper's observation definition rather than this added reference feedback.

[SuperMimic v1](https://arxiv.org/pdf/2610.28480v1#page=13), Appendix A, Table A.1
(PDF page 13), lists only joint positions, joint velocities, base angular velocity,
projected gravity and previous action: `29 + 29 + 3 + 3 + 29 = 93` values.
The reference-relative rotation is absent from that list. However, Table A.2
(PDF page 14) explicitly includes root orientation tracking reward. Removing the
extra actor input therefore does not justify removing that reward.

## Decision

Remove the reference-relative rotation from the actor observation, together with
its optional configuration, six noise entries and reference argument in the pure
proprioception function. Use the same 93-value observation in physical collection,
reconstructed replay histories, imagined rollouts, play and evaluation.

Retain root orientation reward (weight 0.5, scale 0.4), local body orientation reward,
future reference orientation in the goal, and tilt-based termination. The critic
retains its privileged reference-relative orientation feature. Table A.1 specifies
the actor observation; it does not define the critic's feature encoding. Its exact
encoding is still an independent implementation choice.

The resulting default dimensions are:

| Data | Dimension |
|---|---:|
| Actor observation | 93 |
| Ten-frame observation history | 930 |
| Ten-frame goal | 730 |
| Motion latent | 32 |
| Policy decoder input | 962 |
| Packed physical state | 71 |
| World model input / output | 133 / 70 |
| Current critic features / full input | 184 / 914 |

Require checkpoint task schema version 2 and matching task dimensions. Reject
version 1 and checkpoints without task metadata before loading weights. Do not
silently slice old policy weights or normalize legacy input shapes.

## Consequences

Policy observations no longer depend on the selected reference frame. Reference
information reaches the actor through the goal encoder. The reward and critic can
still use full reference state. Maximum per-step reward remains 0.1, preserving
the existing evaluation normalization and return-sampling scale.

The previous 99-value policies and their reported results belong to the historical
implementation; they cannot be loaded into the current task. Training must produce
new checkpoints for the 93-value contract. Verification of shapes and gradients
does not demonstrate tracking performance after this change.

This aligns the published actor observation definition, not every implementation
detail. In particular, the 73-value encoding of each future reference frame remains
independently implemented, and the paper does not specify enough detail to verify
exact equivalence with the authors' goal encoder inputs.
