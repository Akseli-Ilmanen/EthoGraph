import numpy as np

from ethograph.features.changepoints import correct_changepoints_automatic
from ethograph.utils.label_intervals import add_interval, empty_intervals


def test_correct_changepoints_automatic_only_purges_and_stitches():
    df = add_interval(empty_intervals(), 0.0, 0.0005, 1, "bird1")
    df = add_interval(df, 1.0, 2.0, 2, "bird1")
    df = add_interval(df, 2.02, 3.0, 2, "bird1")

    result = correct_changepoints_automatic(
        df,
        min_duration_s=1e-3,
        stitch_gap_s=0.05,
    )

    assert len(result) == 1
    assert result.iloc[0]["labels"] == 2
    np.testing.assert_allclose(result.iloc[0][["onset_s", "offset_s"]], [1.0, 3.0])