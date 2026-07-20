"""An ephys-only drop takes its session timing from the recording itself."""

from pathlib import Path

import numpy as np
import pynwb
import pytest

from ethograph.io.data_loader import _ephys_session_duration, wizard_single_from_ephys

SAMPLE_RATE = 30000.0
LAST_SPIKE = 900000  # samples → 30 s


@pytest.fixture
def kilosort_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "kilosort4"
    folder.mkdir()
    np.save(folder / "spike_times.npy", np.array([0, LAST_SPIKE], dtype=np.int64))
    (folder / "params.py").write_text(f"dat_path = r'F:\\gone\\amplifier.dat'\nsample_rate = {SAMPLE_RATE}\n")
    return folder


def test_duration_from_kilosort_folder(kilosort_folder):
    assert _ephys_session_duration(None, kilosort_folder) == pytest.approx(LAST_SPIKE / SAMPLE_RATE)


def test_no_time_source_raises(tmp_path):
    with pytest.raises(ValueError, match="no readable duration"):
        _ephys_session_duration(None, tmp_path)


def test_wizard_writes_a_trial_spanning_the_recording(kilosort_folder, tmp_path):
    """Regression: this used to fail because trial timing was inferred from
    video/audio/pose columns, which an ephys-only session has none of."""
    wizard_single_from_ephys(output_nc_path=str(tmp_path / "session.nc"), neurons_path=kilosort_folder)

    with pynwb.NWBHDF5IO(str(tmp_path / ".ethograph" / "alignment.nwb"), "r") as io:
        trials = io.read().trials.to_dataframe()

    assert len(trials) == 1
    assert trials["start_time"].iloc[0] == 0.0
    assert trials["stop_time"].iloc[0] == pytest.approx(LAST_SPIKE / SAMPLE_RATE)
