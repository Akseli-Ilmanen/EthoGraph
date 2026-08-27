"""The multi-scale gated shift: starts as the identity, reaches as far as told.

A module that perturbs a pretrained backbone at step 0, or whose reach is a
frame count that silently means something else at another rate, would fail
in ways nobody would trace back here — so those are the two things tested.
"""

from __future__ import annotations

import pytest
import torch

from ethograph.spot.config import ModelConfig
from ethograph.spot.msagsm import MultiScaleGatedShift, _shift


class TestShiftScales:
    def test_the_papers_frames_at_25_fps(self):
        assert ModelConfig().shift_dilations(25.0) == [1, 2, 3]

    def test_same_durations_on_the_strided_200_fps_clock(self):
        # stride 2 -> 100 Hz: 40/80/120 ms = 4/8/12 frames
        assert ModelConfig().shift_dilations(100.0) == [4, 8, 12]

    def test_sub_frame_scales_collapse(self):
        assert ModelConfig(shift_scales_ms=[1.0, 2.0, 40.0]).shift_dilations(25.0) == [1]

    def test_multiscale_is_read_off_the_architecture_name(self):
        assert ModelConfig(architecture="rny008_msagsm").multiscale
        assert not ModelConfig(architecture="rny008_gsm").multiscale


class TestModule:
    def _x(self, t=6, c=16, hw=5, b=2):
        torch.manual_seed(0)
        return torch.randn(b * t, c, hw, hw)

    def test_starts_as_the_identity_up_to_the_attentions_open_scale(self):
        """Zero gates: no frame mixes with another. The only change at step 0
        is the attention's uniform sigmoid(4) ~= 0.982 — a scale, not a shift."""
        m = MultiScaleGatedShift(16, num_segments=6, dilations=(1, 2, 3))
        x = self._x()
        scale = torch.sigmoid(torch.tensor(4.0))
        assert torch.allclose(m(x), scale * x, atol=1e-5)

    def test_shape_is_preserved(self):
        m = MultiScaleGatedShift(16, num_segments=6, dilations=(1, 2))
        x = self._x()
        assert m(x).shape == x.shape

    def test_a_branch_reaches_its_dilation(self):
        """With the gate forced open, a dilation-3 branch moves frame t to t+3."""
        m = MultiScaleGatedShift(16, num_segments=8, dilations=(3,), attention_groups=1)
        with torch.no_grad():
            nn_gate = m.gates[0]
            nn_gate.bias.fill_(10.0)  # tanh -> 1: everything gated, all of it shifted
            m.attention.bias.fill_(50.0)  # sigmoid -> 1
        x = torch.zeros(8, 16, 3, 3)
        x[2] = 1.0  # one frame lit, batch of one
        y = m(x)
        lit = [int(i) for i in torch.nonzero(y.abs().sum(dim=(1, 2, 3))).flatten()]
        assert lit == [5]  # forward half moved +3 ... and the backward half -3 is below 0, so gone

    def test_channels_must_split_into_halves_and_groups(self):
        with pytest.raises(ValueError, match="attention groups"):
            MultiScaleGatedShift(6, num_segments=4, attention_groups=2)

    def test_dilations_must_be_distinct_positive(self):
        with pytest.raises(ValueError, match="distinct positive"):
            MultiScaleGatedShift(16, num_segments=4, dilations=(1, 1))

    def test_shift_helper_zero_pads_both_ways(self):
        x = torch.arange(1.0, 5.0).view(1, 1, 4, 1, 1)
        assert _shift(x, 1).flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
        assert _shift(x, -1).flatten().tolist() == [2.0, 3.0, 4.0, 0.0]
        assert _shift(x, 9).abs().sum() == 0
