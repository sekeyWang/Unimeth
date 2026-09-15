"""Progress helpers for inference."""

from __future__ import annotations


def format_compact_count(value: int) -> str:
    """Format a non-negative counter compactly for a progress bar."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(value)


def _as_int(value) -> int:
    """Convert collated scalar metadata to an integer."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one scalar value, got {value!r}")
        value = value[0]
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


class RecordCompletionTracker:
    """Count records once all of their inference patches have been processed."""

    def __init__(self) -> None:
        self._pending: dict[object, tuple[int, set[int]]] = {}

    @property
    def pending_records(self) -> int:
        return len(self._pending)

    def update_batch(
        self,
        output_record_keys,
        patch_indices,
        total_patches,
    ) -> int:
        keys = list(output_record_keys)
        indices = list(patch_indices)
        totals = list(total_patches)
        if not (len(keys) == len(indices) == len(totals)):
            raise ValueError("Record progress metadata lengths do not match")

        completed = 0
        for record_key, patch_index, patch_total in zip(
            keys,
            indices,
            totals,
        ):
            patch_index = _as_int(patch_index)
            patch_total = _as_int(patch_total)
            if patch_total < 1:
                raise ValueError("total_patches must be at least 1")
            if patch_index < 0 or patch_index >= patch_total:
                raise ValueError(
                    f"patch_idx {patch_index} is outside [0, {patch_total})"
                )

            state = self._pending.get(record_key)
            if state is None:
                received: set[int] = set()
                self._pending[record_key] = (patch_total, received)
            else:
                expected, received = state
                if expected != patch_total:
                    raise ValueError(
                        f"Record {record_key!r} changed total_patches from "
                        f"{expected} to {patch_total}"
                    )

            received.add(patch_index)
            if len(received) == patch_total:
                del self._pending[record_key]
                completed += 1

        return completed
