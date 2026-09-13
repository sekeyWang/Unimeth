"""Inference-only length binning and tensor collation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class InferenceBinning:
    """Group inference patches by signal length to reduce padding."""

    def __init__(self, args):
        self.use_binning = bool(getattr(args, "use_binning", 1))
        self.num_bins = int(args.num_bins)
        self.bin_size = int(args.bin_size)
        self.max_bin_length = int(args.max_bin_length)
        self.bins: list[list[dict[str, Any]]] = [
            [] for _ in range(self.num_bins)
        ]

    def get_data(self, sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Buffer one patch and yield patches from a full length bin."""
        signal_length = len(sample["signals"])
        if signal_length < 50 or signal_length >= self.bin_size * self.num_bins:
            return
        if not self.use_binning:
            yield sample
            return

        bin_id = signal_length // self.bin_size
        selected_bin = self.bins[bin_id]
        selected_bin.append(sample)
        if len(selected_bin) >= self.max_bin_length:
            yield from selected_bin
            selected_bin.clear()

    def _flush_all(self) -> Iterable[dict[str, Any]]:
        """Yield all buffered patches ordered by signal length."""
        pending = []
        for selected_bin in self.bins:
            pending.extend(selected_bin)
            selected_bin.clear()
        pending.sort(key=lambda sample: len(sample["signals"]))
        yield from pending

    def flush(self) -> Iterable[dict[str, Any]]:
        """Yield every patch still buffered at end of input."""
        if self.use_binning:
            yield from self._flush_all()


def collate_inference(
    samples: list[dict[str, Any]],
    total_stride: int = 4,
) -> dict[str, Any]:
    """Pad one inference batch and retain per-record output metadata."""
    if not samples:
        raise ValueError("cannot collate an empty inference batch")

    import torch

    signals = [
        torch.tensor(sample["signals"], dtype=torch.float)
        for sample in samples
    ]
    padded_signals = torch.nn.utils.rnn.pad_sequence(
        signals,
        batch_first=True,
        padding_value=0,
    )

    decoder_inputs = [
        torch.tensor(sample["decoder_input_ids"], dtype=torch.long)
        for sample in samples
    ]
    padded_decoder_inputs = torch.nn.utils.rnn.pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=-100,
    )

    signal_positions = [
        torch.tensor(sample["signal_pos"], dtype=torch.int)
        for sample in samples
    ]
    padded_signal_positions = torch.nn.utils.rnn.pad_sequence(
        signal_positions,
        batch_first=True,
        padding_value=256,
    )

    encoder_masks = [
        torch.ones(
            ((len(sample["signals"]) - 1) // total_stride) + 1,
            dtype=torch.long,
        )
        for sample in samples
    ]
    padded_encoder_masks = torch.nn.utils.rnn.pad_sequence(
        encoder_masks,
        batch_first=True,
        padding_value=0,
    )

    return {
        "signals": padded_signals,
        "encoder_mask": padded_encoder_masks,
        "decoder_input_ids": padded_decoder_inputs,
        "signal_pos": padded_signal_positions,
        "patch_pos": [sample["patch_pos"] for sample in samples],
        "read_pos": [sample["read_pos"] for sample in samples],
        "ref_pos": [sample["ref_pos"] for sample in samples],
        "labels": [sample["labels"] for sample in samples],
        "read_id": [sample["read_id"] for sample in samples],
        "signal_read_id": [
            sample.get("signal_read_id", sample["read_id"])
            for sample in samples
        ],
        "output_record_key": [
            sample.get("output_record_key", sample["read_id"])
            for sample in samples
        ],
        "chr": [sample["chr"] for sample in samples],
        "strand": [sample["strand"] for sample in samples],
        "patch_idx": [sample["patch_idx"] for sample in samples],
        "total_patches": [sample["total_patches"] for sample in samples],
    }
