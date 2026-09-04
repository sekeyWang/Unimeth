"""
Resume helpers for inference.
"""
import re
import shutil
import signal
from pathlib import Path


_ATTEMPT_RE_TEMPLATE = r"rank{rank}\.attempt(\d+)\.started$"


def _as_int(value) -> int:
    """Convert scalar-like batch values to int."""
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def resume_dir_for(output_path: str | Path) -> Path:
    """Return the sidecar resume directory for a final output path."""
    return Path(f"{Path(output_path)}.resume")


class GracefulStopRequested(Exception):
    """Raised internally after SIGTERM/SIGINT requests a resumable stop."""

    def __init__(self, signum=None):
        self.signum = signum
        signal_name = signal.Signals(signum).name if signum is not None else "signal"
        super().__init__(f"graceful stop requested by {signal_name}")


class GracefulStopper:
    """Convert SIGTERM/SIGINT into a stop request checked at safe points."""

    def __init__(self, enabled: bool = True, signals=None):
        self.enabled = enabled
        self.signals = signals or (signal.SIGTERM, signal.SIGINT)
        self.stop_requested = False
        self.signum = None
        self._previous_handlers = {}

    def __enter__(self):
        if not self.enabled:
            return self

        for signum in self.signals:
            try:
                self._previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self.request_stop)
            except (AttributeError, OSError, ValueError):
                continue
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for signum, handler in self._previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (AttributeError, OSError, ValueError):
                continue
        return False

    def request_stop(self, signum, frame):
        self.stop_requested = True
        self.signum = signum

    def raise_if_requested(self):
        if self.stop_requested:
            raise GracefulStopRequested(self.signum)


def _next_attempt_id(resume_dir: Path, rank: int) -> int:
    pattern = re.compile(_ATTEMPT_RE_TEMPLATE.format(rank=rank))
    max_attempt = -1
    if resume_dir.exists():
        for path in resume_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                max_attempt = max(max_attempt, int(match.group(1)))
    return max_attempt + 1


class ResumeCheckpoint:
    """Cumulative completed-read checkpoint for one inference rank."""

    def __init__(self, output_path: str | Path, rank: int):
        self.output_path = Path(output_path)
        self.rank = rank
        self.resume_dir = resume_dir_for(self.output_path)
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        self.completed_path = self.resume_dir / f"rank{rank}.completed_reads.txt"
        self.completed_read_ids = self._load_completed_read_ids()
        self._handle = None

        attempt_id = _next_attempt_id(self.resume_dir, rank)
        self.part_suffix = f"resume{attempt_id}"
        (self.resume_dir / f"rank{rank}.attempt{attempt_id}.started").touch()

    def _load_completed_read_ids(self) -> set[str]:
        completed = set()
        if not self.resume_dir.exists():
            return completed

        for path in sorted(self.resume_dir.glob("rank*.completed_reads.txt")):
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    read_id = line.strip()
                    if read_id:
                        completed.add(read_id)
        return completed

    def refresh_completed_read_ids(self) -> set[str]:
        """Reload completed reads written by all ranks."""
        self.completed_read_ids.update(self._load_completed_read_ids())
        return self.completed_read_ids

    def record_read(self, read_id: str) -> bool:
        """Record one newly completed read ID."""
        if read_id in self.completed_read_ids:
            return False

        if self._handle is None:
            self._handle = open(self.completed_path, "a", encoding="utf-8", buffering=1)

        self.completed_read_ids.add(read_id)
        self._handle.write(f"{read_id}\n")
        self._handle.flush()
        return True

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def cleanup(self):
        self.close()
        if self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)


class ReadCompletionTracker:
    """Track patch completion and write completed read IDs to a checkpoint."""

    def __init__(self, checkpoint: ResumeCheckpoint):
        self.checkpoint = checkpoint
        self._seen_patch_ids: dict[str, set[int]] = {}
        self._expected_patches: dict[str, int] = {}

    def update_batch(self, batch: dict) -> list[str]:
        """Update read completion from a collated inference batch."""
        completed = []
        read_ids = batch.get("read_id", [])
        patch_idx = batch.get("patch_idx", [])
        total_patches = batch.get("total_patches", [])

        for read_id, patch_value, total_value in zip(read_ids, patch_idx, total_patches):
            if read_id in self.checkpoint.completed_read_ids:
                continue

            patch = _as_int(patch_value)
            expected = _as_int(total_value)
            self._expected_patches[read_id] = expected
            seen = self._seen_patch_ids.setdefault(read_id, set())
            seen.add(patch)

            if len(seen) >= expected:
                if self.checkpoint.record_read(read_id):
                    completed.append(read_id)
                self._seen_patch_ids.pop(read_id, None)
                self._expected_patches.pop(read_id, None)

        return completed


def is_completed_tsv_line(line: str, completed_read_ids: set[str]) -> bool:
    """Return whether a TSV line belongs to a completed read."""
    fields = line.split("\t", 5)
    return len(fields) >= 5 and fields[4] in completed_read_ids
