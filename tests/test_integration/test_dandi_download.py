"""Test DANDI download via the wizard's download logic."""

import tempfile
from pathlib import Path

import pytest
from dandi.download import DownloadExisting, download

DANDISET_ID = "001771"
ASSET_PATH = "sub-Neuropixels1-Rat1/sub-Neuropixels1-Rat1_ses-2026-02-13-1_image.nwb"
EXPECTED_FILENAME = "sub-Neuropixels1-Rat1_ses-2026-02-13-1_image.nwb"


@pytest.mark.slow
def test_dandi_download_single_session():
    """Download a single NWB file and verify it lands at the expected path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        url = f"dandi://dandi/{DANDISET_ID}@draft/{ASSET_PATH}"
        download(url, tmpdir, existing=DownloadExisting.SKIP)

        downloaded = Path(tmpdir) / EXPECTED_FILENAME
        assert downloaded.exists(), (
            f"Expected {downloaded} but found: {[p.name for p in Path(tmpdir).rglob('*') if p.is_file()]}"
        )
        assert downloaded.stat().st_size > 1_000_000, "File too small, likely incomplete"

        import h5py

        with h5py.File(downloaded, "r") as f:
            assert "processing" in f, "Not a valid NWB file"
