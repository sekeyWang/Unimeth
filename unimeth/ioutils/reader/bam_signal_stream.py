"""Pair filtered BAM records with raw signals in small batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from unimeth.data.extract import SignalSequenceMismatchError


@dataclass
class BamSignalFeatureStreamStats:
    """Counters collected between BAM filtering and dataset patching."""

    lookup_batches: int = 0
    signal_missing_records: int = 0
    hard_clipped_reconciled_records: int = 0
    signal_sequence_mismatch_records: int = 0
    feature_empty_records: int = 0
    feature_records: int = 0


def _batched(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class BamSignalFeatureStream:
    """Yield feature dictionaries keyed by individual BAM records."""

    def __init__(self, bam_items, signal_lookup, extractor, batch_size: int = 256):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.bam_items = bam_items
        self.signal_lookup = signal_lookup
        self.extractor = extractor
        self.batch_size = batch_size
        self.stats = BamSignalFeatureStreamStats()

    def __iter__(self) -> Iterator[dict]:
        self.stats = BamSignalFeatureStreamStats()
        for bam_batch in _batched(self.bam_items, self.batch_size):
            signal_batch = self.signal_lookup.get_batch(
                item.signal_read_id for item in bam_batch
            )
            self.stats.lookup_batches += 1

            for item in bam_batch:
                signal_read = signal_batch.reads.get(item.signal_read_id)
                if signal_read is None:
                    self.stats.signal_missing_records += 1
                    continue

                try:
                    feature = self.extractor.get_feature(item.bam_record, signal_read)
                except SignalSequenceMismatchError:
                    self.stats.signal_sequence_mismatch_records += 1
                    continue
                if feature is None:
                    self.stats.feature_empty_records += 1
                    continue

                feature = dict(feature)
                if feature.pop("hard_clipped_reconciled", False):
                    self.stats.hard_clipped_reconciled_records += 1
                feature["signal_read_id"] = item.signal_read_id
                feature["output_record_key"] = item.output_record_key
                self.stats.feature_records += 1
                yield feature
