# Reference visit 43 EPA warning and distributed stall

On 2026-10-27, the reference experiment stopped producing metrics after iteration
2150, on visit 43. Both workers were alive; GPU 0 showed 100% utilization and GPU 1
0%, with about 21/20 GiB allocated. The console ended with two warnings:

```text
Warning: EPA horizon = 24 isn't large enough.
```

This was not the earlier CUDA OOM. The last valid checkpoint was `model_2000.pt`.
Original processes were already gone when the diagnosis started; no original
Python stack or GPU-state dump was retained. We did not stop or restart that run.

## Evidence and limits

MuJoCo-Warp 3.11's convex EPA implementation sets the `EPA_HORIZON` overflow bit
when a collision pair's boundary-edge scratch array is exhausted. This capacity
is independent of `nconmax`, `njmax`, and the policy imagination horizon.
The Replay adapter rejects any such overflow after stepping, before recording
the transition or resetting episodes.

The runner previously let a rank-local physics exception escape immediately.
Its peer then entered a WM normalizer `all_reduce`, while the failing worker
entered `destroy_process_group` from the CLI's `finally` block. A controlled
two-A100 reproduction injected the same EPA exception in rank 1's actual
`SuperMimicRunner.collect` step. Stack dumps confirmed rank 0 waiting in the
normalizer collective and rank 1 in NCCL destruction. The diagnostic timed out
after 30 seconds. This reproduces the asymmetric error/cleanup deadlock, not the
original geometric contact itself.

The relevant scenes were:

- rank 0, scene 72/83: Israeli Bezalel carved walnut chair, clip 000.
- rank 1, scene 35/83: New Mexico Pueblo adobe-inspired wooden chair, clip 001.

Separate 512-environment rollouts using `model_2000.pt` completed 150 steps for each
scene without the original warning. An additional 200-step reset stress test also
passed. An unmodified-code continuation from 2000 through 2152 (with fresh Replay)
passed too. Replay, RNG and simulator state are not checkpointed, so none of these
is a bitwise reproduction of the lost original 2150 state. The source/console
evidence and injected-failure test strongly support the two-stage explanation;
they do not prove the exact original collision pair or rule out all other issues.

## Changes

`distributed.local_phase` wraps scene preparation and physical stepping. Every
rank exchanges a success flag before proceeding to the next training collective.
On failure all ranks gather the error text and raise an error containing the
failing rank and original cause. No failed transition is appended to Replay.
Single-process execution preserves the original exception. This adds one small
reduction per physical step and per collection preparation, plus error-detail
gathering only on failure. The wrapped region must not itself contain collectives.
Hung kernels or a destroyed CUDA context still require external timeout handling;
this is not general CUDA failure recovery or automatic checkpoint saving.

Entity scenes now reserve at least 64 EPA boundary edges per convex collision
pair, versus the backend's 24. `EntityEnvCfg.epa_horizon_capacity` accepts 24–256;
it is a process-wide lower bound and never shrinks a larger previous allocation.
The adjustment happens during lazy scene construction, before graph capture.
MuJoCo-Warp 3.11 exposes no public config field for this array, so the integration
sets its pinned-version `collision_convex.MJ_MAX_EPAHORIZON` allocation constant.
No installed dependency file, collision mesh, reward or network was modified.
Existing captured graphs retain their original allocation; a new run is required.

For 512 worlds and 1152 collision capacity, increasing 24 to 64 costs about90 MiB for
one horizon scratch allocation (512×1152×40×4 bytes), before graph reuse/overhead.
It does not guarantee that every future contact fits 64 edges. Overflow detection
remains enabled and must not be suppressed.

## Validation

- Two-process Gloo test fails before the synchronization fix and passes afterward,
  covering rank 1 physics failure, rank 0 scene preparation failure, no rejected
  Replay write, and a usable group for symmetric cleanup.
- Two-A100 NCCL injected-failure repro changes from a timeout to immediate errors
  on both ranks followed by clean shutdown (exit 0 for the catching diagnostic).
- Real near-parallel convex mesh boxes reproduce EPA overflow with an intentionally
  undersized 5-edge array. The entity scene callback expands it and the same
  collision then completes with contacts and no overflow. This tests the backend
  allocation and contact behavior, not merely a config value; it is not a replay
  of the original 24-edge chair overflow.
- Patched two-A100 diagnostic loads the original 2000 checkpoint, deliberately
  selects visit 43, and runs 3 updates with 512 environments/rank, batch 256, 32-step
  imagination, BF16 and compilation. This skip is diagnostic only. All model,
  normalizer and optimizer hashes match across ranks and are finite; no EPA,
  OOM or other physics overflow. Its checkpoint is not a substitute for the
  original experiment's proper resume point.
- Full local suite: 237 passed, including concurrent deployment tests in the
  workspace. Original simulation and distributed regressions remain passing.

Artifacts are under `logs/cuhk_epa_visit43_20261027/` on CUHK and locally.
Server production code and original training outputs were not overwritten; only
the independent diagnostic snapshot received patches. Test GPU allocations were
released. Resume the formal experiment from its own `model_2000.pt` after syncing
the fix. Use 8000 additional iterations to finish at 10000, with the same split,
environment count and scheduling settings.
