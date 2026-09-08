"""Filesystem coordination for uneven multi-process inference workloads."""

import json
import os
import shutil
import time
import uuid
from pathlib import Path


def completion_root_for(output_path: str | Path) -> Path:
    """Return the sidecar directory used to coordinate one output's ranks."""
    return Path(f"{Path(output_path)}.coordination")


def _atomic_write(path: Path, contents: str) -> None:
    """Publish a marker only after its contents are fully written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


class CompletionCoordinator:
    """Coordinate normal rank completion without a tail-end NCCL barrier."""

    def __init__(self, session_dir: str | Path, rank: int, num_processes: int):
        self.session_dir = Path(session_dir)
        self.rank = rank
        self.num_processes = num_processes
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def start(
        cls,
        output_path: str | Path,
        rank: int,
        num_processes: int,
        is_main_process: bool,
        startup_barrier,
    ) -> "CompletionCoordinator":
        """Create one launch-specific session, synchronized before data loading."""
        root = completion_root_for(output_path)
        manifest_path = root / "active.json"

        if is_main_process:
            session_id = uuid.uuid4().hex
            session_dir = root / f"run-{session_id}"
            session_dir.mkdir(parents=True, exist_ok=False)
            _atomic_write(manifest_path, json.dumps({"session_id": session_id}))

        # This is deliberately the only NCCL synchronization in inference control
        # flow. Every rank reaches it before opening files or starting workers.
        startup_barrier()

        for _ in range(100):
            try:
                session_id = json.loads(manifest_path.read_text(encoding="utf-8"))["session_id"]
                return cls(root / f"run-{session_id}", rank, num_processes)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                time.sleep(0.05)

        raise RuntimeError(f"Could not read inference coordination manifest: {manifest_path}")

    def _rank_marker(self, rank: int) -> Path:
        return self.session_dir / f"rank{rank}.done"

    @property
    def _final_marker(self) -> Path:
        return self.session_dir / "final.json"

    def _finalization_ack_marker(self, rank: int) -> Path:
        return self.session_dir / f"rank{rank}.finalized"

    def mark_rank_complete(
        self,
        rank: int | None = None,
        incomplete_read_count: int = 0,
    ) -> None:
        """Publish that a rank has closed all of its output writers."""
        completed_rank = self.rank if rank is None else rank
        _atomic_write(
            self._rank_marker(completed_rank),
            json.dumps(
                {
                    "rank": completed_rank,
                    "incomplete_read_count": int(incomplete_read_count),
                }
            ),
        )

    def wait_for_all(self, poll_interval: float = 1.0, stop_checker=None) -> list[int]:
        """Wait until every rank has safely closed its part output."""
        expected = list(range(self.num_processes))
        while True:
            if stop_checker is not None:
                stop_checker()
            completed = [rank for rank in expected if self._rank_marker(rank).exists()]
            if len(completed) == len(expected):
                return completed
            time.sleep(poll_interval)

    def incomplete_read_count(self) -> int:
        """Return the total incomplete-read count reported by all ranks."""
        total = 0
        for rank in range(self.num_processes):
            marker_path = self._rank_marker(rank)
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                total += int(marker["incomplete_read_count"])
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    f"Invalid inference completion marker: {marker_path}"
                ) from exc
        return total

    def mark_finalized(self, success: bool) -> None:
        """Publish the main rank's final merge status."""
        _atomic_write(self._final_marker, json.dumps({"success": bool(success)}))

    def wait_for_finalization(self, poll_interval: float = 1.0, stop_checker=None) -> bool:
        """Wait for rank 0 to publish its final merge result."""
        while True:
            if stop_checker is not None:
                stop_checker()
            try:
                return bool(json.loads(self._final_marker.read_text(encoding="utf-8"))["success"])
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                time.sleep(poll_interval)

    def acknowledge_finalization(self) -> None:
        """Confirm that this non-main rank observed the finalization result."""
        _atomic_write(self._finalization_ack_marker(self.rank), "")

    def wait_for_finalization_acknowledgements(
        self,
        poll_interval: float = 0.05,
    ) -> None:
        """Wait until every non-main rank has observed the final status."""
        expected = [rank for rank in range(self.num_processes) if rank != self.rank]
        while not all(self._finalization_ack_marker(rank).exists() for rank in expected):
            time.sleep(poll_interval)

    def cleanup(self) -> None:
        """Remove this output's coordination state after successful finalization."""
        root = self.session_dir.parent
        manifest_path = root / "active.json"
        try:
            session_id = json.loads(manifest_path.read_text(encoding="utf-8"))["session_id"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return

        if self.session_dir.name == f"run-{session_id}" and root.exists():
            shutil.rmtree(root)
