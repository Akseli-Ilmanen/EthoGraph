# From an audio file

Use this path for acoustic or vocal data, with or without video.

Supported formats: `.wav`, `.mp3`, `.mp4`, `.flac`. If your `.mp4` video contains audio, point both fields at the same file.

The **Create dialog** handles single recordings. For multiple separate microphone files or multiple trials, see [Multi-trial setup](loading-script.md).

---

## Steps

--8<-- "snippets/launch-single.md"

1. Under **Single trial**, select: **3) Generate from audio file**
2. Click **Next →** — the dialog opens
3. Set **Audio file** (`.wav`, `.mp3`, `.mp4`, `.flac`)
4. Optionally set **Video file** — frame rate is auto-detected; audio sample rate is read-only (auto-detected)
5. Set **Output path** for the generated `trials.nc`
6. Click **Generate .nc file**
7. The I/O widget auto-populates → click **Load**

---

## No-video mode

When no video is provided EthoGraph enters **no-video mode**: a time slider replaces the napari video player, and playback uses `sounddevice`. All labelling and changepoint features work the same way.

---

## Multichannel audio

If all microphones are stored in a **single multichannel `.wav`** file, load it directly — EthoGraph separates channels automatically.

For **multiple separate `.wav` files** (one per mic), use the multi-trial wizard: see [Multi-trial setup — session-wide audio](loading-script.md#session-wide-audio).

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Video file** | Optional |
| **Audio file** | `.wav`, `.mp3`, `.mp4`, `.flac` |
| **Video frame rate** | Auto-detected from video |
| **Video onset in audio** | Time offset (s) where video starts relative to audio stream |
| **Audio sample rate** | Read-only — auto-detected from file |
| **Individuals** | Optional, comma-separated |
| **Load video motion features** | Extracts frame-to-frame motion signal from video |
| **Output path** | Where to save the generated `trials.nc` |

---

## Data requirements

| Attribute | Value |
|-----------|-------|
| `attrs["fps"]` | Not required for audio-only datasets |
| `coords["individuals"]` | Optional |

See [Data Format Requirements — Audio only](data-requirements.md#audio-only-no-video).
