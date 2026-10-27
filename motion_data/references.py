"""One reference per chair, selected only from an existing manual keep list."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from .selection import SelectionStore


def chair_id(filename):
    parts = Path(filename).stem.split("__")
    if len(parts) != 3 or parts[0] != "sitting" or not parts[1] or not parts[2].isdigit():
        raise ValueError(f"Expected sitting__<chair>__<index>.npz: {filename}")
    return parts[1]


class ChairReferences:
    def __init__(self, selection: SelectionStore, output: Path):
        self.selection = selection
        self.output = Path(output).resolve()
        if self.output == selection.output_json:
            raise ValueError("Reference choices must use a separate file from the original selection")
        self.groups = {}
        for name in selection.files:
            if selection.get(name)["status"] == "keep":
                self.groups.setdefault(chair_id(name), []).append(name)
        self.choices = {}
        if self.output.exists():
            saved = json.loads(self.output.read_text())
            if (saved.get("version") != 1 or saved.get("source_dir") != str(selection.source_dir)
                    or saved.get("selection_json") != str(selection.output_json)):
                raise ValueError("Reference file belongs to a different source or selection")
            choices = saved.get("choices")
            if not isinstance(choices, dict):
                raise ValueError("Invalid reference choices")
            for chair, name in choices.items():
                if name not in selection.files or chair_id(name) != chair:
                    raise ValueError(f"Invalid reference for {chair}: {name}")
            self.choices = choices

    def resolved(self, chair):
        candidates = self.groups.get(chair, [])
        if self.choices.get(chair) in candidates:
            return self.choices[chair]
        return candidates[0] if len(candidates) == 1 else None

    def summary(self):
        return dict(chairs=len(self.groups), candidates=sum(map(len, self.groups.values())),
                    single=sum(len(names) == 1 for names in self.groups.values()),
                    resolved=sum(self.resolved(chair) is not None for chair in self.groups),
                    pending=sum(self.resolved(chair) is None for chair in self.groups))

    def choose(self, filename):
        chair = chair_id(filename)
        if filename not in self.groups.get(chair, []):
            raise ValueError("Only manually kept motions may be selected as references")
        self._save({**self.choices, chair: filename})

    def clear(self, chair):
        self._save({key: value for key, value in self.choices.items() if key != chair})

    def _save(self, choices):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self.output.parent, encoding="utf-8",
                                             suffix=".json", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(dict(version=1, source_dir=str(self.selection.source_dir),
                               selection_json=str(self.selection.output_json), choices=choices),
                          stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output)
            self.choices = choices
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    def export(self, destination, mode="symlink"):
        if mode not in ("symlink", "copy"):
            raise ValueError("Unknown export mode")
        pending = self.summary()["pending"]
        if pending or not self.groups:
            raise ValueError(f"还有 {pending} 种椅子待选 reference；完成后再导出。")
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            output = destination / f"references-{index:04d}"
            try:
                output.mkdir()
                break
            except FileExistsError:
                index += 1
        references = {chair: self.resolved(chair) for chair in self.groups}
        try:
            for name in references.values():
                source = self.selection.source_dir / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                if mode == "symlink":
                    (output / name).symlink_to(source)
                else:
                    shutil.copy2(source, output / name)
            (output / "manifest.json").write_text(json.dumps(dict(
                version=1, source_dir=str(self.selection.source_dir),
                selection_json=str(self.selection.output_json), reference_json=str(self.output),
                mode=mode, references=references, candidates=self.groups,
                implicit_single_candidates=[c for c in self.groups if len(self.groups[c]) == 1]),
                ensure_ascii=False, indent=2) + "\n")
            (output / "references.txt").write_text("".join(name + "\n" for name in references.values()))
        except BaseException:
            shutil.rmtree(output)
            raise
        return output
