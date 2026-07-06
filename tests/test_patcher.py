import numpy as np

from unimeth.data.patcher import patch_sequence


def test_patch_sequence_handles_short_sequence_with_methylation_site():
    bases = list("AC") + ["[CpG]"] + list("GTA")
    signal_event = [np.array([1.0], dtype=np.float32) for _ in bases]
    signal_event[2] = np.array([], dtype=np.float32)
    meta_data = [{"ref_pos": 10, "labels": -1, "read_pos": 1}]

    patches = list(
        patch_sequence(
            mode="inference",
            bases=bases,
            signal_event=signal_event,
            shift=0.0,
            scale=1.0,
            meta_data=meta_data,
            chunk_size=256,
            overlap=16,
            total_stride=4,
        )
    )

    assert len(patches) == 1
    assert patches[0]["patch_pos"] == [2]
    assert patches[0]["read_pos"] == [1]
    assert patches[0]["ref_pos"] == [10]
