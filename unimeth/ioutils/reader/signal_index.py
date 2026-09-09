"""Persistent read-ID routing for multi-file raw signal input."""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator


_SCHEMA_VERSION = "1"
_LOOKUP_CHUNK_SIZE = 900
_INSERT_BATCH_SIZE = 10_000


class DuplicateSignalReadIdError(RuntimeError):
    """Raised when one signal read ID cannot be routed unambiguously."""


@dataclass(frozen=True)
class SignalIndexStats:
    """Summary of one signal route index build."""

    index_path: str
    file_count: int
    read_count: int
    elapsed_seconds: float
    reused: bool = False


@dataclass(frozen=True)
class SignalIndexProgress:
    """File-level progress emitted while a route index is built."""

    files_completed: int
    file_count: int
    read_count: int
    current_path: str
    elapsed_seconds: float


@dataclass(frozen=True)
class SignalRoutingPlan:
    """Pickle-safe description of how workers should route signal reads."""

    signal_paths: tuple[str, ...]
    index_path: str | None
    index_stats: SignalIndexStats | None

    @property
    def uses_index(self) -> bool:
        return self.index_path is not None

    def open_router(self) -> "SignalReadRouter":
        return SignalReadRouter(self)


def iter_signal_read_ids(signal_path: str | os.PathLike[str]) -> Iterator[str]:
    """Yield all read IDs from one POD5/SLOW5/BLOW5 file."""
    from unimeth.ioutils.reader.raw_signal import is_pod5_path, is_slow5_path

    if is_pod5_path(signal_path):
        import pod5 as p5

        with p5.Reader(signal_path) as signal_file:
            for read_id in signal_file.read_ids:
                yield str(read_id)
        return

    if is_slow5_path(signal_path):
        from unimeth.ioutils.reader.slow5 import (
            _flatten_read_ids,
            _open_slow5,
        )

        signal_file = _open_slow5(signal_path)
        try:
            for read_id in _flatten_read_ids(signal_file.get_read_ids()):
                yield read_id
        finally:
            close = getattr(signal_file, "close", None)
            if close is not None:
                close()
        return

    raise ValueError(f"Unsupported raw signal file extension: {signal_path}")


def _normalize_signal_paths(
    signal_paths: Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    paths = tuple(Path(path).resolve() for path in signal_paths)
    if not paths:
        raise ValueError("at least one signal file is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Signal file not found: {path}")
    return paths


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE signal_files (
            file_id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL
        );
        CREATE TABLE reads (
            read_id TEXT PRIMARY KEY,
            file_id INTEGER NOT NULL REFERENCES signal_files(file_id)
        ) WITHOUT ROWID;
        """
    )


def _open_read_only(index_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{index_path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _matching_index_counts(
    index_path: Path,
    signal_paths: tuple[Path, ...],
) -> tuple[int, int] | None:
    if not index_path.is_file():
        return None
    try:
        connection = _open_read_only(index_path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if (
                metadata.get("schema_version") != _SCHEMA_VERSION
                or metadata.get("complete") != "1"
            ):
                return None
            stored_files = tuple(
                connection.execute(
                    "SELECT path, size, mtime_ns FROM signal_files ORDER BY file_id"
                )
            )
        finally:
            connection.close()
    except (OSError, sqlite3.DatabaseError, ValueError):
        return None

    current_files = tuple(
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in signal_paths
    )
    if stored_files != current_files:
        return None
    try:
        return int(metadata["file_count"]), int(metadata["read_count"])
    except (KeyError, ValueError):
        return None


def _batched(values: Iterable[tuple[str, int]], size: int):
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _insert_read_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[str, int]],
    current_path: Path,
) -> None:
    """Insert one batch, taking the slower diagnostic path only on collision."""
    connection.execute("SAVEPOINT signal_read_batch")
    try:
        connection.executemany(
            "INSERT INTO reads(read_id, file_id) VALUES (?, ?)",
            batch,
        )
    except sqlite3.IntegrityError:
        connection.execute("ROLLBACK TO signal_read_batch")
        connection.execute("RELEASE signal_read_batch")
    else:
        connection.execute("RELEASE signal_read_batch")
        return

    for read_id, file_id in batch:
        existing = connection.execute(
            "SELECT signal_files.path "
            "FROM reads JOIN signal_files USING (file_id) "
            "WHERE reads.read_id = ?",
            (read_id,),
        ).fetchone()
        if existing is not None:
            raise DuplicateSignalReadIdError(
                f"Signal read ID {read_id!r} occurs more than once: "
                f"{existing[0]} and {current_path}"
            )
        connection.execute(
            "INSERT INTO reads(read_id, file_id) VALUES (?, ?)",
            (read_id, file_id),
        )


def build_signal_route_index(
    signal_paths: Iterable[str | os.PathLike[str]],
    index_path: str | os.PathLike[str],
    force_rebuild: bool = False,
    progress_callback: Callable[[SignalIndexProgress], None] | None = None,
) -> SignalIndexStats:
    """Build a complete SQLite ``signal_read_id -> signal file`` index."""
    started_at = time.monotonic()
    paths = _normalize_signal_paths(signal_paths)
    destination = Path(index_path).resolve()
    if not force_rebuild:
        counts = _matching_index_counts(destination, paths)
        if counts is not None:
            file_count, read_count = counts
            return SignalIndexStats(
                index_path=str(destination),
                file_count=file_count,
                read_count=read_count,
                elapsed_seconds=time.monotonic() - started_at,
                reused=True,
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(f"{destination.name}.building")
    if building.exists():
        building.unlink()

    connection = sqlite3.connect(str(building))
    read_count = 0
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        _create_schema(connection)

        for file_id, path in enumerate(paths, start=1):
            stat = path.stat()
            connection.execute(
                "INSERT INTO signal_files(file_id, path, size, mtime_ns) "
                "VALUES (?, ?, ?, ?)",
                (file_id, str(path), stat.st_size, stat.st_mtime_ns),
            )
            rows = ((read_id, file_id) for read_id in iter_signal_read_ids(path))
            for batch in _batched(rows, _INSERT_BATCH_SIZE):
                _insert_read_batch(connection, batch, path)
                read_count += len(batch)
            if progress_callback is not None:
                progress_callback(
                    SignalIndexProgress(
                        files_completed=file_id,
                        file_count=len(paths),
                        read_count=read_count,
                        current_path=str(path),
                        elapsed_seconds=time.monotonic() - started_at,
                    )
                )

        metadata = (
            ("schema_version", _SCHEMA_VERSION),
            ("complete", "1"),
            ("file_count", str(len(paths))),
            ("read_count", str(read_count)),
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata,
        )
        connection.commit()
    except Exception:
        connection.close()
        if building.exists():
            building.unlink()
        raise
    else:
        connection.close()

    os.replace(building, destination)
    return SignalIndexStats(
        index_path=str(destination),
        file_count=len(paths),
        read_count=read_count,
        elapsed_seconds=time.monotonic() - started_at,
    )


def prepare_signal_routing(
    signal_paths: Iterable[str | os.PathLike[str]],
    index_path: str | os.PathLike[str] | None,
    force_rebuild: bool = False,
    progress_callback: Callable[[SignalIndexProgress], None] | None = None,
) -> SignalRoutingPlan:
    """Prepare single-file direct routing or a reusable multi-file index."""
    paths = _normalize_signal_paths(signal_paths)
    normalized_paths = tuple(str(path) for path in paths)
    if len(paths) == 1:
        return SignalRoutingPlan(
            signal_paths=normalized_paths,
            index_path=None,
            index_stats=None,
        )
    if index_path is None:
        raise ValueError("index_path is required for multiple signal files")

    stats = build_signal_route_index(
        paths,
        index_path,
        force_rebuild=force_rebuild,
        progress_callback=progress_callback,
    )
    return SignalRoutingPlan(
        signal_paths=normalized_paths,
        index_path=stats.index_path,
        index_stats=stats,
    )


class SignalRouteIndex:
    """Read-only batch lookup against a completed signal route index."""

    def __init__(self, index_path: str | os.PathLike[str]):
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Signal route index not found: {self.index_path}")

        self._connection = _open_read_only(self.index_path)
        metadata = dict(self._connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema_version") != _SCHEMA_VERSION:
            self.close()
            raise RuntimeError("Unsupported signal route index schema")
        if metadata.get("complete") != "1":
            self.close()
            raise RuntimeError("Signal route index is incomplete")
        self._file_paths = {
            file_id: path
            for file_id, path in self._connection.execute(
                "SELECT file_id, path FROM signal_files"
            )
        }

    def lookup_many(self, read_ids: Iterable[str]) -> dict[str, str]:
        """Return routes for found IDs; missing IDs are omitted."""
        unique_ids = tuple(dict.fromkeys(str(read_id) for read_id in read_ids))
        found: dict[str, str] = {}
        for start in range(0, len(unique_ids), _LOOKUP_CHUNK_SIZE):
            chunk = unique_ids[start : start + _LOOKUP_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"SELECT read_id, file_id FROM reads "
                f"WHERE read_id IN ({placeholders})",
                chunk,
            )
            for read_id, file_id in rows:
                found[read_id] = self._file_paths[file_id]
        return {read_id: found[read_id] for read_id in unique_ids if read_id in found}

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "SignalRouteIndex":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class SignalReadRouter:
    """Route a batch of signal read IDs without a per-worker Python index."""

    def __init__(self, plan: SignalRoutingPlan):
        self.signal_paths = plan.signal_paths
        self._single_path = plan.signal_paths[0] if not plan.uses_index else None
        self._index = SignalRouteIndex(plan.index_path) if plan.uses_index else None

    def lookup_many(self, read_ids: Iterable[str]) -> dict[str, str]:
        unique_ids = tuple(dict.fromkeys(str(read_id) for read_id in read_ids))
        if self._single_path is not None:
            return {read_id: self._single_path for read_id in unique_ids}
        if self._index is None:
            raise RuntimeError("Signal read router is closed")
        return self._index.lookup_many(unique_ids)

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    def __enter__(self) -> "SignalReadRouter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
