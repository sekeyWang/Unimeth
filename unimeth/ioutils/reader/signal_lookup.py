"""Small-batch signal lookup for BAM-primary streaming inference."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


class SignalBatchReader(Protocol):
    def get_many(self, read_ids: list[str]) -> dict[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SignalLookupBatch:
    """Found signal records and the requested IDs that could not be loaded."""

    reads: dict[str, object]
    missing_read_ids: tuple[str, ...]


class _Pod5BatchReader:
    def __init__(self, path: str):
        import pod5 as p5

        self._reader = p5.Reader(path)

    def get_many(self, read_ids: list[str]) -> dict[str, object]:
        records = self._reader.reads(
            selection=read_ids,
            missing_ok=True,
            preload={"samples"},
        )
        return {str(record.read_id): record for record in records}

    def close(self) -> None:
        self._reader.close()


class _Slow5BatchReader:
    def __init__(self, path: str):
        from unimeth.ioutils.reader.slow5 import Slow5Reader

        self._reader = Slow5Reader(path, load_read_ids=False)

    def get_many(self, read_ids: list[str]) -> dict[str, object]:
        return self._reader.get_reads(read_ids)

    def close(self) -> None:
        self._reader.close()


def _open_signal_batch_reader(path: str) -> SignalBatchReader:
    suffix = Path(path).suffix.lower()
    if suffix == ".pod5":
        return _Pod5BatchReader(path)
    if suffix in {".slow5", ".blow5"}:
        return _Slow5BatchReader(path)
    raise ValueError(f"Unsupported raw signal file extension: {path}")


class SignalBatchLookup:
    """Fetch routed signals in per-file batches with a bounded reader cache."""

    def __init__(
        self,
        router,
        max_open_files: int = 4,
        reader_factory: Callable[[str], SignalBatchReader] | None = None,
    ):
        if max_open_files < 1:
            raise ValueError("max_open_files must be >= 1")
        self.router = router
        self.max_open_files = max_open_files
        self._reader_factory = reader_factory or _open_signal_batch_reader
        self._readers: OrderedDict[str, SignalBatchReader] = OrderedDict()

    def _get_reader(self, path: str) -> SignalBatchReader:
        reader = self._readers.pop(path, None)
        if reader is not None:
            self._readers[path] = reader
            return reader

        while len(self._readers) >= self.max_open_files:
            _, expired = self._readers.popitem(last=False)
            expired.close()
        reader = self._reader_factory(path)
        self._readers[path] = reader
        return reader

    def get_batch(self, read_ids: Iterable[str]) -> SignalLookupBatch:
        requested = tuple(dict.fromkeys(str(read_id) for read_id in read_ids))
        routes = self.router.lookup_many(requested)
        grouped: dict[str, list[str]] = defaultdict(list)
        for read_id in requested:
            path = routes.get(read_id)
            if path is not None:
                grouped[path].append(read_id)

        found: dict[str, object] = {}
        for path, routed_ids in grouped.items():
            found.update(self._get_reader(path).get_many(routed_ids))

        reads = {read_id: found[read_id] for read_id in requested if read_id in found}
        missing = tuple(read_id for read_id in requested if read_id not in found)
        return SignalLookupBatch(reads=reads, missing_read_ids=missing)

    def close(self) -> None:
        while self._readers:
            _, reader = self._readers.popitem(last=False)
            reader.close()
        close_router = getattr(self.router, "close", None)
        if close_router is not None:
            close_router()

    def __enter__(self) -> "SignalBatchLookup":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
