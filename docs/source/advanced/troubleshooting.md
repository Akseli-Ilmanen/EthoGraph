(target-troubleshooting)=
# Troubleshooting

Report bugs on [GitHub Issues](https://github.com/Akseli-Ilmanen/ethograph/issues).

---

## Quick fixes

| Problem | Solution |
|---------|----------|
| Unexpected error in the GUI | Save labels (`Ctrl + S`), then restart the GUI. Save semi-regularly! |
| Error during data loading | Click **Reset gui_settings.yaml** in the I/O widget to reset the state of the GUI. |

---

## FAQ

### My dataset format is not supported

I/O support for new data formats is actively being expanded. If your format is not yet represented, please send a sample dataset to [akseli.ilmanen@gmail.com](mailto:akseli.ilmanen@gmail.com) and I will work on adding loading support for it.


### What does this button do?

A lot of buttons, spinners, dropdowns, etc. have a little description tooltip when you hover over them.

### Installation fails with "resolution-too-deep"

This happens when using plain `pip` to install ethograph with extras like
`[gui]` or `[all]`. The dependency tree (pygfx + movement + pynwb) is too
complex for pip's resolver.

**Fix:** Use `uv` instead of `pip`:

```bash
uv pip install "ethograph[all]"
```

See {doc}`../getting_started/installation` for full instructions.

(no-audio-device)=
### Silent audio on Linux

If playback is silent, check what PortAudio sees:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

A default device of `[-1, -1]` means PortAudio found no device. On a
PipeWire/PulseAudio desktop this is almost always the missing ALSA bridge —
install it and restart ethograph:

```bash
sudo apt install libasound2-plugins pipewire-alsa
```

(Debian/Ubuntu's bundled PortAudio only speaks ALSA, so it needs this bridge
to reach a modern sound server.) Routing playback through `pw-play`/`paplay`
instead makes audio audible but not sample-accurately synced, so it isn't
suitable for precise annotation.

(audio-formats)=
### Supported audio formats

The waveform, spectrogram and playback all read audio through
[`audioio`](https://github.com/bendalab/audioio) (libsndfile). Supported
**standalone** audio files:

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV    | `.wav`    | Recommended — PCM `.wav` loads fastest and is the best-tested path. |
| FLAC   | `.flac`   | Lossless, compact. |
| OGG    | `.ogg`    | Vorbis. |
| MP3    | `.mp3`    | Requires a recent libsndfile (≥ 1.1). |

**Audio embedded in a video (`.mp4`/`.mov`/`.avi`) is not read in place.** The
AAC/other codecs inside a video container are not decoded for analysis, so these
extensions are deliberately absent from the audio file picker. How to get the
audio out depends on how you load:

- **Drag & drop:** when you drop a video that has an embedded audio track, tick
  **"extract audio"** on the cover page — EthoGraph extracts it to a throwaway
  `.wav` that then feeds the normal `audio_mic-N` pipeline. This is a
  convenience for one-off, single-session loads.
- **Custom set-up:** there is no per-video extraction step here. Convert your
  videos to audio **in bulk yourself** before loading (one WAV/FLAC/OGG per
  clip), then point the loader at those files. For example, with `ffmpeg`:

  ```bash
  for f in *.mp4; do ffmpeg -i "$f" -vn -acodec pcm_s16le "${f%.mp4}.wav"; done
  ```

  Extracting once, up front, is faster and more reliable than re-decoding the
  container on every load.

If a file cannot be decoded you get a single log line naming the file and the
reason (e.g. *"Cannot load audio from video container … enable 'extract audio'"*)
rather than a stream of errors; convert the file to one of the formats above.

### Opening `.tsv` label files in Excel

Excel on Windows may not correctly parse `.tsv` files when double-clicked due to regional delimiter settings.

**Automatic fix:** EthoGraph automatically registers `.tsv` files to open correctly in Excel with tab delimiters the first time you run it. On Windows this writes to the current-user registry (no admin prompt); on macOS it uses `duti` if installed.

If the association is not working, you can re-run it manually:

```python
from ethograph.utils.download import ensure_default_configs
ensure_default_configs()
```

**Manual alternative:**

1. Open Excel -> **File -> Open -> Browse**
2. Change file filter to **"All Files (\*.\*)"**
3. Select the `.tsv` file
4. In the Text Import Wizard, select **Tab** as delimiter
