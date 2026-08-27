"""Multi-scale attention gated shift — E2E-Spot's GSM with a wider aperture.

The Gate Shift Module (Sudhakaran et al. 2020) mixes each frame with its two
neighbours: half the channels are gated and shifted one frame forward, the
other half one frame back. At 25 fps that is ±40 ms of context inside the
backbone; at 200 fps it is ±5 ms, and the stride ladder showed that buying
aperture with ``stride`` runs out of road once the label grid coarsens past
10 ms. MSAGSM (Liu et al. 2025, *Multi-Scale Attention Gated Shifting for
Precise Event Spotting*, arXiv 2507.07381) widens the aperture **without
touching the label grid**: the gated shift runs at several temporal dilations
at once and the branches are blended by learned softmax weights, after a
channel-grouped spatial attention has weighted the frame.

Written from the paper on top of the BSD-2 GSM that E2E-Spot vendors — the
reference MSAGSM repository carries no licence. Three choices the paper
leaves to the reader are made here on purpose and are worth knowing:

* **Branch weights are a softmax**, as the paper states. Summed raw, a module
  whose gates start at zero would return ``len(dilations) × x``.
* **The module starts as the identity, up to a uniform scale.** Gates are
  zero-initialised (GSM's own rule) so no frame mixes with another, and the
  attention's bias starts at 4 so its sigmoid is ≈ 0.98 everywhere — a scale
  the next BatchNorm absorbs, with the gradient still alive. A pretrained
  backbone is not perturbed at step 0; the module has to earn its effect.
* **Every dilation is a count of *strided* frames**, resolved by the config
  from milliseconds (:meth:`~ethograph.spot.config.ModelConfig.shift_dilations`).
  The paper's ``{1, 2, 3}`` are 25 fps numbers.

Called from the vendored ``model/shift.py`` in place of ``_GSM`` — same
``(channels, n_segment)`` constructor, same ``(B·T, C, H, W)`` in and out.
"""

from __future__ import annotations

import torch
from torch import nn

#: Sigmoid(4) ≈ 0.982: the attention starts almost fully open.
_ATTENTION_OPEN_BIAS = 4.0


def _shift(x: torch.Tensor, frames: int) -> torch.Tensor:
    """Shift ``(B, C, T, H, W)`` along T by *frames* (negative = backwards), zero padded."""
    out = torch.zeros_like(x)
    t = x.shape[2]
    if abs(frames) >= t:
        return out
    if frames > 0:
        out[:, :, frames:] = x[:, :, :-frames]
    elif frames < 0:
        out[:, :, :frames] = x[:, :, -frames:]
    else:
        out = x
    return out


class MultiScaleGatedShift(nn.Module):
    """GSM at several temporal dilations, softmax-blended, behind grouped attention.

    *channels* must be divisible by ``2 * attention_groups``: the gated shift
    splits the channels into a forward and a backward half, and the attention
    into ``attention_groups`` equal slices.
    """

    def __init__(
        self,
        channels: int,
        num_segments: int,
        dilations: tuple[int, ...] | list[int] = (1, 2, 3),
        attention_groups: int = 2,
    ) -> None:
        super().__init__()
        dilations = tuple(int(d) for d in dilations)
        if not dilations or any(d < 1 for d in dilations) or len(set(dilations)) != len(dilations):
            raise ValueError(f"dilations must be distinct positive frame counts, got {dilations}")
        if attention_groups < 1 or channels % (2 * attention_groups):
            raise ValueError(f"{channels} channels cannot be split into 2 halves x {attention_groups} attention groups")
        self.channels = channels
        self.num_segments = num_segments
        self.dilations = dilations
        self.attention_groups = attention_groups

        # Channel-grouped spatial attention: one map per group, over the frame.
        self.attention = nn.Conv2d(channels, attention_groups, kernel_size=3, padding=1)
        nn.init.zeros_(self.attention.weight)
        nn.init.constant_(self.attention.bias, _ATTENTION_OPEN_BIAS)

        # One gate per dilation, reaching d frames either side; zero-initialised
        # so the gated part is nothing and the shift is a no-op at the start.
        self.gates = nn.ModuleList()
        for d in dilations:
            gate = nn.Conv3d(channels, 2, kernel_size=3, padding=(d, 1, 1), dilation=(d, 1, 1), groups=2)
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
            self.gates.append(gate)
        self.norm = nn.BatchNorm3d(channels)
        self.branch_logits = nn.Parameter(torch.zeros(len(dilations)))

    def _gated_shift(self, x: torch.Tensor, gate_conv: nn.Conv3d, frames: int) -> torch.Tensor:
        """GSM's split-gate-shift-residual at one dilation, on ``(B, C, T, H, W)``."""
        gate = torch.tanh(gate_conv(torch.relu(self.norm(x))))
        half = self.channels // 2
        first, second = x[:, :half], x[:, half:]
        gated_first = gate[:, 0:1] * first
        gated_second = gate[:, 1:2] * second
        forward = _shift(gated_first, frames) + (first - gated_first)
        backward = _shift(gated_second, -frames) + (second - gated_second)
        return torch.cat([forward, backward], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0] // self.num_segments
        c, h, w = x.shape[1:]
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        # Attention on each frame, one map per channel group.
        maps = torch.sigmoid(self.attention(x))
        per_group = c // self.attention_groups
        x = x * maps.repeat_interleave(per_group, dim=1)
        # (B·T, C, H, W) -> (B, C, T, H, W) for the temporal shifts.
        x = x.view(batch, self.num_segments, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        weights = torch.softmax(self.branch_logits, dim=0)
        y = sum(w * self._gated_shift(x, gate, d) for w, gate, d in zip(weights, self.gates, self.dilations))
        return y.permute(0, 2, 1, 3, 4).contiguous().view(batch * self.num_segments, c, h, w)
