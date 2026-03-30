"""Functional test for WindowedBuffer + FileSource + downsampling."""

import numpy as np

from ethograph.gui.modality import FileSource, SourceData, WindowedBuffer
from ethograph.gui.plots_audiotrace import _downsample_minmax


class MockLoader:
    def __init__(self, n_samples, rate):
        self.rate = rate
        self._data = np.sin(2 * np.pi * 440 * np.arange(n_samples) / rate).astype(
            np.float64
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]


loader = MockLoader(441000, 44100.0)
source = FileSource("audio", loader, start_time=0.0)

print(
    f"Source: {source.name}, range={source.time_range}, "
    f"sr={source.sampling_rate}, id={source.identity}"
)
assert source.time_range.start_s == 0.0
assert abs(source.time_range.end_s - 10.0) < 0.001

data = source.get_data(1.0, 2.0)
print(
    f"get_data(1, 2): {len(data)} samples, "
    f"t=[{data.timestamps[0]:.4f}, {data.timestamps[-1]:.4f}]"
)
assert len(data) > 44000

buf = WindowedBuffer(buffer_multiplier=3.0)
buf.set_source(source)

result = buf.get(2.0, 3.0)
assert result is not None
print(f"Buffer loaded: {len(result)} samples, cache_range={buf.cache_range}")
assert buf.cache_range[0] < 2.0
assert buf.cache_range[1] > 3.0

old_range = buf.cache_range
result2 = buf.get(2.2, 2.8)
assert buf.cache_range == old_range
print("Cache hit OK")

result3 = buf.get(8.0, 9.0)
assert buf.cache_range != old_range
print(f"Reloaded for pan: cache_range={buf.cache_range}")

source2 = FileSource("audio", loader, start_time=0.0, channel=0)
buf.set_source(source2)
assert buf._cache is None
print("Identity invalidation OK")

big_data = source.get_data(0.0, 5.0)
times, amps, step = _downsample_minmax(big_data, 0.0, 5.0, 800)
print(f"Downsample: {len(big_data)} -> {len(times)} points, step={step}")
assert step > 1
assert len(times) == len(amps)
assert times[0] >= 0.0
assert times[-1] <= 5.0

small_data = source.get_data(0.0, 0.005)
times_s, amps_s, step_s = _downsample_minmax(small_data, 0.0, 0.005, 800)
print(f"No downsample: {len(small_data)} -> {len(times_s)} points, step={step_s}")
assert step_s == 1

print()
print("ALL TESTS PASSED")
