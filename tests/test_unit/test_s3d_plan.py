"""S3D extraction is configured in seconds; frames come from the video's rate."""

import pytest

from ethograph.video_features.plan import MIN_STACK, S3DConfig, plan_s3d


@pytest.mark.parametrize(
    "video_fps, analysis_fps, stack_s, step, stack_frames",
    [
        (200.0, None, 0.1, 1, 21),  # today's setting, every frame
        (200.0, 25.0, 0.5, 8, 13),  # subsample to 25 fps; 0.5 s = 12.5 → odd
        (60.0, 25.0, 0.4, 2, 13),  # 60 → 30 fps
        (30.0, 25.0, 0.5, 1, 15),  # never upsamples: step stays 1
        (30.0, None, 0.5, 1, 15),
        (25.0, 50.0, 0.6, 1, 15),  # asking for more than the video has → every frame
    ],
)
def test_plan_table(video_fps, analysis_fps, stack_s, step, stack_frames):
    plan = plan_s3d(video_fps, S3DConfig(analysis_fps=analysis_fps, stack_s=stack_s))
    assert (plan.step, plan.stack_frames) == (step, stack_frames)
    assert plan.effective_fps == pytest.approx(video_fps / step)
    assert plan.stack_frames % 2 == 1
    assert plan.stack_frames >= MIN_STACK
    assert f"step {step}" in plan.describe()


def test_a_window_too_short_for_the_rate_is_refused_with_the_minimum():
    """0.1 s is 21 frames at 200 fps but 3 at 30 fps — the plan says what would work."""
    with pytest.raises(ValueError, match=r"stack_s >= 0\.433 s"):
        plan_s3d(30.0, S3DConfig(stack_s=0.1))
    with pytest.raises(ValueError, match="higher analysis_fps"):
        plan_s3d(200.0, S3DConfig(analysis_fps=25.0, stack_s=0.1))


def test_rates_are_never_defaulted():
    with pytest.raises(ValueError, match="read it from the video"):
        plan_s3d(0.0, S3DConfig())
    with pytest.raises(ValueError, match="analysis_fps must be positive"):
        plan_s3d(30.0, S3DConfig(analysis_fps=0.0))


def test_stack_seconds_round_trips():
    plan = plan_s3d(200.0, S3DConfig(stack_s=0.1))
    assert plan.stack_s == pytest.approx(21 / 200)
