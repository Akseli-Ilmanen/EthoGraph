# Playback & the timeline marker

## The red marker is on *time*, not the video frame

Video is recorded as discrete frames (e.g. ~47 per second), but audio and other
signals have far finer time resolution — a birdsong syllable can be shorter than
a single video frame. So the red timeline marker (the playhead) tracks the true
**time**, not the nearest video frame.

When you click on a plot (waveform, spectrogram, line plot):

- the marker jumps to **exactly where you clicked**, down to sub-frame precision;
- the video seeks to the **nearest frame** to that time and displays it;
- a label you place is stored at the **exact clicked time**, not snapped to a frame.

```{note}
Because the video can only show whole frames, the displayed frame and the red
marker can sit up to half a frame apart (~10 ms). This is expected and lets you
place a syllable boundary precisely between two frames — you judge the boundary
from the waveform/spectrogram, and the video is a reference. Stepping frame by
frame (arrow keys) moves the marker onto exact frame times.
```

## Playback modes

The **playback mode** dropdown in the bottom bar picks how video (and audio) play:

- **Audio-synced** — audio and video are locked together, driven by the audio
  output itself, so they never drift. Under load the video may drop frames to
  stay in sync. This is the only mode that plays sound.
- **Smooth (every frame)** — shows every video frame in order; under load it
  runs slower than the set FPS rather than skipping frames. No audio.
- **Real-time (skip frames)** — approximates the set FPS by skipping frames when
  needed. No audio.

The default follows your data (audio present → Audio-synced, otherwise Smooth);
once you pick a mode it is remembered. Playback speed is the **FPS** field next
to the dropdown — audio pitch is normal when FPS equals the recording rate.

## Which channel Play will sound

When you press **Play** (or Space), the video and the selected audio play
together. With several audio panels open, playback follows the **last audio
selection you made**, so you always hear what you are looking at:

- clicking an audio trace / spectrogram panel, or
- changing that panel's **Channel** dropdown in the right sidebar (shown when an
  audio panel is active).

The bottom bar shows a small speaker icon with the active channel as
`ChN: filename…`; hover over it to see the full channel name.

```{note}
During playback the marker is driven by the **audio output itself** (an
audio-master clock), so the playhead, the video frame and the sound stay locked
and don't drift apart over long clips. If no audio device is available, playback
falls back to a frame timer; above ~2.5× speed the audio is dropped (it would be
unintelligible) and the video free-runs.
```
