"""Persistent manual motion decisions and non-destructive subset exports."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import RLock


STATUSES = ("unreviewed", "keep", "reject", "later")


class SelectionStore:
    """Autosave decisions for immediate ``*.npz`` files in one source directory.

    Tags are arbitrary nonempty strings, for example ``back_contact`` or
    ``unclear``. A saved source path must match before labels can be resumed.
    Original motion files are never modified.
    """

    def __init__(self, source_dir: str | Path, output_json: str | Path):
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.output_json = Path(output_json).expanduser().resolve()
        if not self.source_dir.is_dir():
            raise ValueError(f"Motion source is not a directory: {self.source_dir}")
        if self.output_json.suffix.lower() != ".json":
            raise ValueError("Selection output must be a .json file")
        self.files = tuple(sorted(p.name for p in self.source_dir.glob("*.npz") if p.is_file()))
        self._names = frozenset(self.files)
        self._lock = RLock()
        for filename in self.files:
            self._validate_filename(filename)
        self._decisions = {name: self._record("unreviewed", [], "") for name in self.files}
        if self.output_json.exists():
            try:
                saved = json.loads(self.output_json.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot read selection file: {self.output_json}") from error
            if not isinstance(saved, dict) or saved.get("version") != 1:
                raise ValueError("Unsupported selection file format")
            if saved.get("source_dir") != str(self.source_dir):
                raise ValueError("Saved selection source directory does not match this source")
            if not isinstance(saved.get("decisions"), dict):
                raise ValueError("Saved selection decisions must be an object")
            for filename, record in saved["decisions"].items():
                self._validate_filename(filename)
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid saved decision for {filename}")
                self._decisions[filename] = self._record(
                    record.get("status"), record.get("tags", []), record.get("notes", "")
                )

    def _validate_filename(self, filename):
        if (not isinstance(filename, str) or not filename
                or any(char in filename for char in ("/", "\\", "\0", "\n", "\r"))
                or filename not in self._names):
            raise ValueError(f"Unknown or unsafe motion filename: {filename!r}")

    @staticmethod
    def _record(status, tags, notes):
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError(f"Status must be one of {STATUSES}")
        if (not isinstance(tags, (list, tuple))
                or any(not isinstance(tag, str) or not tag.strip()
                       or any(ord(char) < 32 for char in tag) for tag in tags)):
            raise ValueError("Tags must be a list of nonempty strings without control characters")
        if not isinstance(notes, str):
            raise ValueError("Notes must be a string")
        return {"status": status, "tags": list(dict.fromkeys(tags)), "notes": notes}

    def get(self, filename: str) -> dict:
        """Return an independent copy of one motion's decision."""
        self._validate_filename(filename)
        with self._lock:
            return deepcopy(self._decisions[filename])

    def set(self, filename: str, status: str, *, tags=None, notes=None) -> dict:
        """Atomically save a decision; omitted tags/notes preserve their old values."""
        self._validate_filename(filename)
        with self._lock:
            previous = self._decisions[filename]
            record = self._record(status, previous["tags"] if tags is None else tags,
                                  previous["notes"] if notes is None else notes)
            decisions = {**self._decisions, filename: record}
            self._save(decisions)
            self._decisions = decisions
            return deepcopy(record)

    def summary(self) -> dict:
        """Return a count for every status, plus ``total``."""
        with self._lock:
            counts = {status: 0 for status in STATUSES}
            for record in self._decisions.values():
                counts[record["status"]] += 1
            return {**counts, "total": len(self.files)}

    def _save(self, decisions):
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=self.output_json.parent,
                                             prefix=".selection-", suffix=".json",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"version": 1, "source_dir": str(self.source_dir),
                           "decisions": decisions}, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output_json)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def export(self, destination: str | Path, mode: str = "symlink") -> Path:
        """Export kept motions into a fresh ``selection-0001`` style directory.

        Includes ``manifest.json`` and ``kept.txt``. Repeated exports retain older
        subsets and cannot carry stale rejected files into a new subset. Failure
        removes only this invocation's partial directory. Symlinks require the
        original data to remain available; copies are standalone.
        """
        if mode not in ("symlink", "copy"):
            raise ValueError("Export mode must be 'symlink' or 'copy'")
        with self._lock:
            decisions = deepcopy(self._decisions)
        kept = [name for name in self.files if decisions[name]["status"] == "keep"]
        if not kept:
            raise ValueError("No kept motions to export")
        destination = Path(destination).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            output = destination / f"selection-{number:04d}"
            try:
                output.mkdir()
                break
            except FileExistsError:
                number += 1
        try:
            for filename in kept:
                source = self.source_dir / filename
                if not source.is_file():
                    raise FileNotFoundError(f"Motion source disappeared: {source}")
                if mode == "symlink":
                    (output / filename).symlink_to(source)
                else:
                    shutil.copy2(source, output / filename)
            manifest = {"version": 1, "source_dir": str(self.source_dir),
                        "selection_json": str(self.output_json), "mode": mode,
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                        "files": kept, "decisions": {name: decisions[name] for name in kept}}
            (output / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (output / "kept.txt").write_text("".join(name + "\n" for name in kept), encoding="utf-8")
        except BaseException:
            shutil.rmtree(output)
            raise
        return output
