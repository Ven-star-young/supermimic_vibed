# Sitting depth student

This task follows the OpenDoor BC → DAgger organization. A frozen sitting expert
labels the state visited by the student; the student receives only robot
proprioception, robot reference and the current depth frame.

## Inputs and timing

| Input | Shape | Encoding |
| --- | --- | --- |
| Robot history | `[B,10,93]`, oldest to newest | RunningNorm → 930 values |
| Robot reference | `[B,730]` | Existing goal encoder → 32 values |
| Current depth | `[B,1,120,160]`, float32 metres | CNN → 64 values |

The 93 robot channels match OpenDoor: joint positions29, joint velocities29,
body angular velocity3, projected gravity3, previous action29. The student's
actor receives 1026 values and produces the existing 29 normalized joint
actions with unchanged joint order and action scale. It does not receive chair
pose, geometry IDs, CAD features or tactile observations.

The depth CNN follows HDMI: three `5×5`, stride2, padding2 convolutions with
eight channels and Mish activations. Shapes are
`1×120×160 → 8×60×80 → 8×30×40 → 8×15×20`;
flattening gives 2400 values, followed by `Linear(2400,64)` and `LayerNorm(64)`.
This follows the official HDMI [convolution helper](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/learning/ppo/common.py#L74)
and [depth estimator](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/learning/ppo/ppo_roa.py#L274).
The existing goal encoder and actor hidden widths are `2048,1024,512,256`.
The goal encoder starts from the teacher and remains trainable.

The shared G1 asset holds the HDMI camera pose/intrinsics. Native mjlab sensing
captures a fresh frame every 20 ms control step, after four 5 ms physics steps.
These are **simulation-time frequencies**: training need not run in real time.
Depth is optical-axis Z distance; nonfinite values are cleaned and values are
clipped to `0.1–4 m`. Images bypass RunningNorm, with no additional resize,
noise, delay, frame stacking or recurrent state in this baseline.
Warp's zero no-return pixels map to 4 m before clipping, as in the deployment
adapter. This preserves the distinction between missing depth and a near obstacle.

Depth, robot history, reference and teacher label all describe the same
pre-action state. The collector snapshots them before stepping, records the
transition's done flag, then resets completed environments. RSI remains
adaptive within each environment's assigned chair. Reset refreshes geometry
and depth; robot history uses the existing first-frame backfill.
Completed episodes update the adaptive sampler with discounted return and the
existing timeout bootstrap. Returns persist across BC/DAgger collection rounds
and reset only for environments whose episodes actually end.

## Run

From the repository root, check the teacher schema, original chair split,
training NPZ hashes and collision caches without starting simulation:

```bash
uv run python -m supermimic.scripts.distill_sitting --check
```

Train with the A100 40 GiB settings in `configs/sitting_dagger_depth.json`
(use the Slurm command below on CUHK):

```bash
uv run python -m supermimic.scripts.distill_sitting
```

For another output directory or rollout/batch sizes, copy the JSON and pass
`--config /absolute/path/to/config.json`. The output directory must be empty.
`num_envs` must cover all training chair/scale variants; each world keeps a
fixed chair and samples only that chair's reference clips. Training uses the
same split fingerprint and collision profile as the teacher.
The JSON uses 664 environments (eight per training chair) and batch size 512.
For the local 8 GiB GPU, use a separate JSON with `num_envs=83` and
`batch_size=64`; 128 simultaneous environments exceeded that GPU during scene
graph initialization. The checked-in JSON is sized for the server.

The teacher is `logs/cuhk_sitting_deployment/model_10000.pt`. BC first collects
teacher-only transitions. DAgger then chooses teacher or student actions per
environment according to beta, while always storing the teacher's label.
The CNN and actor train together on action MSE; teacher parameters and
normalizers remain frozen and are checked after every round.

### CUHK and the selected chairs

Use the expert's existing `sitting_back_references_seed0.json` split: 83 selected
training chairs with one reference each, and 22 held-out chairs for evaluation.
The checkpoint's saved split and this manifest have the same fingerprint.
`data_root` may contain additional motions; collection loads only the training
references listed in the manifest. Changing `num_envs` changes replication of
these chairs, not membership, and must leave at least one world per chair.

The CUHK checkout is `/home/wangfeng/supermimic_vibed`; its teacher is
`logs/sitting_task_2gpu/model_10000.pt`. The shared JSON retains the local teacher
path. `tools/distill_sitting.sbatch` overrides only the teacher path, output
directory and device in a saved run configuration, following the OpenDoor
submission layout. It requests one A100, eight CPUs and 64 GiB host memory,
checks teacher/split identity and training file hashes, then starts one learner.
Slurm selects the physical GPU; the learner uses its visible `cuda:0`.

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/slurm
sbatch tools/distill_sitting.sbatch
```

Each submission writes to `logs/sitting_dagger_depth/slurm-<job_id>/` and logs
to `logs/slurm/sitting-depth-distill-<job_id>.out`. It does not overwrite an
earlier run. The entrypoint uses one GPU/process; the older two-GPU expert
`torchrun` command does not apply to DAgger.

| Setting | Single A100 baseline |
| --- | --- |
| Environments / training batch | 664 / 512 |
| BC collection | 256 steps × 664 = 169,984 labelled samples |
| BC optimization | 4 × 332 batches = 1,328 gradient updates |
| DAgger rounds / collection per round | 32 / 64 steps × 664 = 42,496 new samples |
| Optimization per round | 4 × 128 batches = 512 gradient updates |
| Teacher action probability | 1 → 0 over 16 rounds; last 16 rounds use student actions |
| Learning rate / gradient norm limit | 0.0002 / 1.0 |

Sampling is with replacement: `train_epochs × updates_per_round` is the actual
number of gradient updates, not a count of complete dataset traversals. Each
round samples 262,144 transitions for optimization, matching the full buffer
size. BC samples four times its collected data count. This is an initial
training budget; held-out task success determines whether more training is
needed.

### No-depth ablation

The matched proprioception baseline is `configs/sitting_dagger_proprio.json`.
It changes only `use_depth` to `false` and writes to
`logs/sitting_dagger_proprio`; all environment, split, RSI, reference, BC,
DAgger, beta schedule, buffer, batch and optimizer settings are identical to
the depth configuration. The student input is the OpenDoor-style 930-D robot
history plus the 32-D goal latent (962 actor inputs), with no CNN, camera
sensor, image transfer or depth ring-buffer allocation. Its checkpoint kind is
`sitting_proprio_student`.

Run the matched ablation on CUHK with:

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/slurm
sbatch tools/distill_sitting.sbatch configs/sitting_dagger_proprio.json
```

The output directory is made unique by Slurm job ID, so the result is
`logs/sitting_dagger_proprio/slurm-<job_id>/`.

The CUHK A100 short check passed with 664 environments across all 83 chairs,
teacher and student rollouts, episode resets and batch-512 gradient updates.
Observed GPU use with batch 512 was about 33.35 GiB. Thirty-two teacher steps
took 9.27 s (about 20% more samples/second than the 332-environment check).
The student completed 128 steps with 1,303 episode resets in 39.60 s; sixteen
batch-512 updates took 0.74 s. Scene startup took about 4.4 minutes. These are
short-run resource measurements, not evidence of task convergence.

Depth is held in a bounded CPU ring buffer as float16 and converted back to
float32 for training. The 262,144-transition ring occupies about 11.02 GiB of
host memory and retains roughly 6.17 rounds of new samples. Its 16 GiB
`max_dataset_gib` limit rejects accidental oversized buffers. GPU rendering,
models and sampled batches require additional memory.

Outputs include `config.json`, the unchanged split membership in `split.json`,
`train_log.csv`, `model_bc.pt` and `model_<round>.pt`. Checkpoints identify
`kind="sitting_depth_student"`, include the student input contract, camera
metadata, teacher path/iteration/digest and chair experiment provenance.
The `task` field retains the teacher's physics/reward schema for environment
validation; `kind` and `policy_config` define the student's observation schema.
The no-depth baseline uses `kind="sitting_proprio_student"` and omits camera
metadata.

## Evaluation

For a Slurm run, replace `logs/sitting_dagger_depth/model_32.pt` in the examples
below with `logs/sitting_dagger_depth/slurm-<job_id>/model_32.pt`.

After training, view the first training reference with the student:

```bash
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_dagger_depth/model_32.pt \
  --data-root data/motions/sitting --viewer native --episodes 3
```

Evaluate all held-out references from frame zero, with the existing clean and
perturbed starts and full reference/stable-contact metrics:

```bash
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_dagger_depth/model_32.pt \
  --data-root data/motions/sitting --partition test --evaluate \
  --viewer none --episodes 3 --output /tmp/sitting_depth_test.json
```

Repeat the same evaluation with `--depth-input constant` and a different
`--output` path to replace every depth image with 4 m for a visual ablation.

Use held-out chairs from the saved split, without adding them to DAgger's
training buffer. Evaluate complete clips from frame zero in addition to RSI
rollouts. Compare normal images against fixed or shuffled depth to check
whether the student uses vision; action MSE alone is not a task success metric.
Single-frame vision can miss chair geometry at mid-motion RSI starts; this
baseline intentionally has no visual memory to reconstruct unseen surfaces.

This student uses supervised BC/DAgger only. The existing sitting world model
does not predict future depth, so this implementation does not feed the vision
student into MBRL imagination.

## Export

Export the trained student, including its CNN and normalizers:

```bash
uv run python -m supermimic.scripts.export_sitting \
  --checkpoint logs/sitting_dagger_depth/model_32.pt \
  --output /tmp/sitting_depth.onnx
```

The three inputs remain proprioceptive history, robot reference and processed
float32 depth. Camera acquisition and the `0.1–4 m` depth preprocessing belong
to the runtime; the ONNX output retains the task's 29 normalized actions.

The companion `unitree-sd` sitting task selects this ONNX with
`UNITREE_SITTING_DEPTH_POLICY=/absolute/path/sitting_depth.onnx`; see its sitting
README for the MuJoCo sim2sim command and camera display. Its packaged chair and
reference are retained, while the student does not consume the expert's baked
CAD feature or simulated chair pose.

## Inspect camera output

The existing kinematic replay tool uses the same depth cleaning and resamples
camera replay to 50 Hz. This inspects reference visibility, not policy success:

```bash
uv run python -m tools.replay_motion data/motions/sitting --index 0 --sensor-view
```

`--sensor-output /tmp/sitting_depth.npz` records RGB `[T,120,160,3]`, processed
`depth [T,1,120,160]`, raw `distance_to_image_plane [T,120,160,1]`, frame IDs and
simulation timestamps. Raw depth is retained separately from the clipped input.
