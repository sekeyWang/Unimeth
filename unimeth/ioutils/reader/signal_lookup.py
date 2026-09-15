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
    """Signals keyed by BAM record plus requests that could not be resolved."""

    reads: dict[int | str, object]
    missing_read_ids: tuple[str, ...]
    missing_record_keys: tuple[int | str, ...] = ()
    missing_source_hint_record_keys: tuple[int | str, ...] = ()


@dataclass(frozen=True)
class SignalLookupRequest:
    """One BAM record's request for its parent raw signal."""

    output_record_key: int | str
    signal_read_id: str
    signal_source_hint: str | None = None


class SignalSourceHintError(RuntimeError):
    """Raised when a BAM fn tag cannot select one signal-file candidate."""


def _source_basename(value: str) -> str:
    """Normalize BAM fn values and local paths to a portable basename."""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


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

    def get_batch(
        self,
        requests: Iterable[SignalLookupRequest | str],
    ) -> SignalLookupBatch:
        normalized = []
        seen_record_keys = set()
        for request in requests:
            is_lookup_request = isinstance(request, SignalLookupRequest)
            if is_lookup_request:
                normalized_request = request
            else:
                read_id = str(request)
                normalized_request = SignalLookupRequest(
                    output_record_key=read_id,
                    signal_read_id=read_id,
                )
            if normalized_request.output_record_key in seen_record_keys:
                if not is_lookup_request:
                    continue
                raise ValueError(
                    "output_record_key must be unique within a signal lookup batch"
                )
            seen_record_keys.add(normalized_request.output_record_key)
            normalized.append(normalized_request)

        signal_read_ids = tuple(
            dict.fromkeys(request.signal_read_id for request in normalized)
        )
        lookup_candidates = getattr(self.router, "lookup_candidates_many", None)
        if lookup_candidates is None:
            routes = self.router.lookup_many(signal_read_ids)
            candidates = {
                read_id: (path,)
                for read_id, path in routes.items()
            }
        else:
            candidates = lookup_candidates(signal_read_ids)

        grouped: dict[str, list[SignalLookupRequest]] = defaultdict(list)
        missing_record_keys = []
        missing_read_ids = []
        missing_source_hint_record_keys = []
        for request in normalized:
            paths = candidates.get(request.signal_read_id, ())
            if not paths:
                missing_record_keys.append(request.output_record_key)
                missing_read_ids.append(request.signal_read_id)
                continue
            if len(paths) == 1:
                grouped[paths[0]].append(request)
                continue

            source_hint = request.signal_source_hint
            if not source_hint:
                missing_source_hint_record_keys.append(request.output_record_key)
                continue
            hint_basename = _source_basename(source_hint)
            matches = tuple(
                path for path in paths
                if _source_basename(path) == hint_basename
            )
            if len(matches) != 1:
                raise SignalSourceHintError(
                    f"BAM fn tag {source_hint!r} cannot uniquely route signal "
                    f"read ID {request.signal_read_id!r}; candidates: "
                    f"{', '.join(paths)}"
                )
            grouped[matches[0]].append(request)

        reads: dict[int | str, object] = {}
        for path, routed_requests in grouped.items():
            routed_ids = list(
                dict.fromkeys(
                    request.signal_read_id
                    for request in routed_requests
                )
            )
            found = self._get_reader(path).get_many(routed_ids)
            for request in routed_requests:
                signal_read = found.get(request.signal_read_id)
                if signal_read is None:
                    missing_record_keys.append(request.output_record_key)
                    missing_read_ids.append(request.signal_read_id)
                else:
                    reads[request.output_record_key] = signal_read

        return SignalLookupBatch(
            reads=reads,
            missing_read_ids=tuple(missing_read_ids),
            missing_record_keys=tuple(missing_record_keys),
            missing_source_hint_record_keys=tuple(
                missing_source_hint_record_keys
            ),
        )

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
