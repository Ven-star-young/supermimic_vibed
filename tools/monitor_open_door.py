"""One idempotent check: archive new CUHK models and replay their policies on Lexar.

Scheduling belongs to the Codex heartbeat. This command performs one bounded pass;
failed downloads/evaluations remain pending and are retried on the next pass.
"""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = "Mjlab-SuperMimic-OpenDoor-Unitree-G1"


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, timeout=600, **kwargs)


def remote(cfg, source):
    result = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                  cfg["host"], "python3 -"], input=source, capture_output=True)
    return json.loads(result.stdout)


def inspect_training(cfg):
    # Paths are Python literals on stdin, never interpolated into shell code.
    return remote(cfg, rf"""
import csv, json, os, re, subprocess
from pathlib import Path
root = Path({cfg['remote_run']!r})
if not root.is_dir(): raise FileNotFoundError(root)
rows = list(csv.DictReader((root / 'train_log.csv').open())) if (root / 'train_log.csv').exists() else []
models = []
for p in root.glob('model_*.pt'):
    m = re.fullmatch(r'model_(\d+)\.pt', p.name)
    if m:
        stat = p.stat()
        models.append(dict(name=p.name, iteration=int(m[1]), size=stat.st_size))
processes = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'supermimic.scripts.train_open_door' in cmd and 'python' in cmd and os.readlink(p / 'cwd') == {cfg['remote_repo']!r}:
            # The launcher holds its torchrunx logs open: bind liveness to this
            # run, rather than another training job in the same repository.
            for fd in (p / 'fd').iterdir():
                try:
                    if os.readlink(fd).startswith(str(root.resolve()) + '/'):
                        processes.append(dict(pid=int(p.name), command=cmd))
                        break
                except OSError: pass
    except (OSError, UnicodeError): pass
agent = (root / 'params/agent.yaml').read_text()
target = int(re.search(r'^max_iterations: (\d+)', agent, re.M)[1])
latest = rows[-1] if rows else {{}}
iteration = int(latest.get('iteration', 0))
complete = iteration >= target and any(m['iteration'] >= target for m in models)
logs = sorted(root.glob('torchrunx/*/localhost*.log'), key=lambda p:p.stat().st_mtime)
errors = []
for p in logs:
    for line in p.read_text(errors='replace').splitlines()[-40:]:
        if any(word in line for word in ('FloatingPointError', 'OutOfMemoryError', 'RuntimeError:', 'NCCL error', 'Nonfinite', 'out of memory')):
            errors.append(line)
gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'], capture_output=True,text=True)
print(json.dumps(dict(run=str(root), state='completed' if complete else 'running' if processes else 'interrupted',
    latest=latest, target=target, processes=processes, models=sorted(models,key=lambda m:m['iteration']),
    log_mtime=(root/'train_log.csv').stat().st_mtime if rows else None,
    errors=errors[-12:], gpu=gpu.stdout)))
""")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(cfg, source, destination, *, verify=False, expected_size=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         f"{cfg['host']}:{source}", str(temporary)], capture_output=True)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise ValueError(f"Incomplete download: {source}")
    checksum = None
    if verify:
        with temporary.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        expected = remote(cfg, "import hashlib,json\nfrom pathlib import Path\n"
                          f"h=hashlib.sha256()\nwith Path({source!r}).open('rb') as f:\n"
                          " for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)\n"
                          "print(json.dumps(h.hexdigest()))")
        if checksum != expected:
            raise ValueError(f"Checksum mismatch: {source}")
    temporary.replace(destination)
    return checksum


def snapshot_reference(cfg, directory):
    manifest = directory / "reference_manifest.json"
    if manifest.exists():
        return
    files = remote(cfg, "import json\nfrom pathlib import Path\n"
                   f"p=Path({cfg['remote_motion_dir']!r})\n"
                   "print(json.dumps([str(f) for f in sorted(p.iterdir()) if f.suffix in ('.npz','.xml')]))")
    if not any(p.endswith('.npz') for p in files):
        raise ValueError("Remote prepared reference contains no NPZ")
    checksums = {}
    for source in files:
        checksums[Path(source).name] = download(cfg, source, directory / "reference" / Path(source).name, verify=True)
    write_json(manifest, {"source": cfg["remote_motion_dir"], "sha256": checksums})


def evaluate_model(cfg, directory, checkpoint):
    folder = directory / "evaluations" / checkpoint.stem
    folder.mkdir(parents=True, exist_ok=True)
    trajectory = folder / "rollout.npz"
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "MUJOCO_GL": "egl"}
    with (folder / "evaluation.log").open('w') as stream:
        run([sys.executable, "-m", "supermimic.scripts.evaluate", "--task", TASK,
             "--checkpoint-file", str(checkpoint), "--motion-dir", str(directory / "reference"),
             "--device", cfg["device"], "--record", str(trajectory)],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    metrics = checkpoint.with_name(checkpoint.stem + "_eval.json")
    report = json.loads(metrics.read_text())
    write_json(folder / "metrics.json", report)
    motions = sorted((directory / "reference").glob('*.npz'))
    if not motions:
        raise ValueError("Missing archived reference motions")
    videos = []
    with (folder / "render.log").open('w') as stream:
        for motion in motions:
            video = folder / f"{motion.stem}.mp4"
            temporary = video.with_name(video.stem + ".partial.mp4")
            run([sys.executable, "-m", "tools.render", str(trajectory), str(temporary),
                 "--clips", motion.stem, "--door-xml", str(motion.with_suffix('.xml')),
                 "--motion-file", str(motion), "--seconds", "60", "--title", checkpoint.stem],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,width,height", str(temporary)], stdout=stream, stderr=subprocess.STDOUT)
            if temporary.stat().st_size == 0:
                raise ValueError(f"Empty video: {temporary}")
            temporary.replace(video)
            videos.append(str(video))
    return {"checkpoint": str(checkpoint), "metrics": str(folder / 'metrics.json'),
            "videos": videos, "iteration": report['iteration'], "norm_return": report['norm_return'],
            **{key: report['door_tracking'][key] for key in
               ('reference_completion_rate', 'angle_rmse_rad', 'angular_velocity_rmse_radps')}}


def check(cfg):
    mount = Path(cfg['mount'])
    if not mount.is_mount():
        raise RuntimeError(f"Lexar is not mounted: {mount}")
    root = Path(cfg['output_root']).resolve()
    if not root.is_relative_to(mount.resolve()):
        raise ValueError("output_root must be on the Lexar mount")
    directory = root / Path(cfg['remote_run']).name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'monitor.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "busy", "new_models": []}
        status = inspect_training(cfg)
        status['checked_at'] = datetime.now(timezone.utc).isoformat()
        write_json(directory / 'status.json', status)
        download(cfg, cfg['remote_run'] + '/train_log.csv', directory / 'train_log.csv')
        for name in ('agent.yaml', 'env.yaml'):
            if not (directory / 'params' / name).exists():
                download(cfg, cfg['remote_run'] + '/params/' + name, directory / 'params' / name)
        snapshot_reference(cfg, directory)
        completed, errors = [], []
        for item in status['models']:
            checkpoint = directory / 'checkpoints' / item['name']
            marker = directory / 'evaluations' / checkpoint.stem / 'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                videos_ready = saved['videos'] and all(
                    Path(p).is_file() and Path(p).stat().st_size > 0 for p in saved['videos'])
                if (checkpoint.exists() and checkpoint.stat().st_size == item['size']
                        and Path(saved['metrics']).is_file() and videos_ready):
                    continue
                marker.unlink()
            try:
                receipt = checkpoint.with_suffix('.download.json')
                if not (checkpoint.exists() and checkpoint.stat().st_size == item['size'] and receipt.exists()):
                    digest = download(cfg, cfg['remote_run'] + '/' + item['name'], checkpoint,
                                      verify=True, expected_size=item['size'])
                    write_json(receipt, {"source": cfg['remote_run'] + '/' + item['name'], "sha256": digest})
                report = evaluate_model(cfg, directory, checkpoint)
                write_json(marker, report)
                completed.append(report)
                print(json.dumps({"completed": report}, ensure_ascii=False), flush=True)
            except (subprocess.SubprocessError, ValueError, OSError, KeyError) as exc:
                errors.append({"checkpoint": item['name'], "error": str(exc)})
        reports = [json.loads(p.read_text()) for p in (directory / 'evaluations').glob('*/complete.json')]
        reports.sort(key=lambda r:r['iteration'])
        fields = ('iteration','norm_return','reference_completion_rate','angle_rmse_rad','angular_velocity_rmse_radps','checkpoint')
        with (directory / 'evaluation_summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(reports)
        result = {"state": status['state'], "iteration": status['latest'].get('iteration'),
                  "directory": str(directory), "new_models": completed, "errors": errors}
        write_json(directory / 'last_check.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/open_door_monitor.json')
    args = parser.parse_args()
    result = check(json.loads(args.config.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get('errors'):
        sys.exit(1)


if __name__ == '__main__':
    main()
