#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and robot description, then build native MJCF.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python - <<'PY'
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.request import urlopen

dataset = "lvhaidong/LAFAN1_Retargeting_Dataset"
api = f"https://huggingface.co/api/datasets/{dataset}"
requested_revision = os.environ.get("LAFAN1_REVISION", "main")
with urlopen(f"{api}/revision/{quote(requested_revision, safe='')}", timeout=60) as response:
    revision = json.load(response)["sha"]

def files(folder):
    with urlopen(f"{api}/tree/{revision}/{folder}", timeout=60) as response:
        return [item["path"] for item in json.load(response) if item["type"] == "file"]

downloads = []
downloads.extend((p, Path("data/lafan1_g1") / Path(p).name) for p in files("g1") if p.endswith(".csv"))
urdf = "robot_description/g1/g1_29dof_rev_1_0.urdf"
downloads.append((urdf, Path("data/g1_description") / Path(urdf).name))
downloads.extend((p, Path("data/g1_description/meshes") / Path(p).name) for p in files("robot_description/g1/meshes"))
for source, target in downloads:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{quote(source)}"
    print(f"Downloading {source}", flush=True)
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
Path("data/source.json").write_text(json.dumps({
    "dataset": dataset, "revision": revision,
    "files": [source for source, _ in downloads],
}, indent=2) + "\n")
print(f"Downloaded {len(list(Path('data/lafan1_g1').glob('*.csv')))} clips at {revision}")
PY
uv run python tools/urdf_to_mjcf.py
