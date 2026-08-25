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

(command-not-found)=
### `ethograph` is not recognized as a command

The terminal reports something like:

```text
The term 'ethograph' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

The install worked — the `ethograph` command just isn't on your `PATH` yet.
`uv tool install` puts it in uv's own bin directory, which uv has to register
with your shell:

```bash
uv tool update-shell
```

Then **close the terminal and open a new one**. `PATH` is only read when a shell
starts, so the current window will keep reporting the same error. After that,
`ethograph launch` works from any directory.

Two alternatives if you would rather not touch `PATH`:

- **A shortcut instead of a command.** `ethograph shortcut` creates a
  desktop/Start Menu entry that launches the GUI on double-click. It writes the
  full interpreter path into the shortcut, so nothing needs to be activated or
  on `PATH`.
- **A one-off launch.** `uvx --from "ethograph[gui,audio]" ethograph launch`
  runs the GUI without installing a command at all.

```{warning}
`uv tool list` reports only **tool** installs. If you have *also* installed
ethograph into a virtual environment or a conda environment, that is a second,
independent copy — often at a different version — and launching through that
environment's `python -m ethograph launch` runs it instead of the one
`uv tool list` shows.

When reporting a bug, take the version from the interpreter you actually
launched with, not from `uv tool list`:

    python -c "import ethograph, sys; print(ethograph.__version__, sys.prefix)"

Keeping a single install avoids the confusion entirely.
```

(linux-missing-libraries)=
### Linux: `libOpenGL.so.0`, "could not load the Qt platform plugin", black video

On Linux the pip wheels load a handful of shared libraries from the
distribution, and each one that is missing fails in its own words:

| You see | What is missing |
|---------|-----------------|
| `OpenGL.platform.ctypesloader \| Failed to load library ( 'libOpenGL.so.0' )` | `libopengl0` (and usually `libgl1`) — PyOpenGL's GLVND dispatch |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | the xcb libraries (`libxcb-cursor0`, `libxkbcommon-x11-0`, …) |
| The GUI opens but the video panel stays black, or wgpu reports no adapter | `libvulkan1` + `mesa-vulkan-drivers` |
| `OSError: PortAudio library not found` | `libportaudio2` (audio extra) |

Run the preflight — it checks all of them at once and prints the install
line for your distribution:

```bash
ethograph check
```

`ethograph launch` prints the same warning at startup, so if a launch ends
in one of the errors above, scroll up: the fix is already on screen. The full
list, per distribution, is in {ref}`linux-system-libraries`.

(wsl)=
### Windows Subsystem for Linux (WSL)

The GUI runs under WSL through **WSLg** (Windows 11, or `wsl --update` from
PowerShell on Windows 10), which provides the display. A WSL distro is a
minimal install, so start with the system libraries above — every one of them
is typically missing — then `ethograph check`.

Two things to know:

- **Video rendering is best-effort.** The video canvas is drawn with wgpu,
  which does not officially support WSL. It works through mesa's software
  (lavapipe) or Microsoft's `dzn` Vulkan driver, both in `mesa-vulkan-drivers`,
  but slower than native. If the video panel stays black after installing
  them, run ethograph natively on Windows — the data on `/mnt/c/...` is the
  same data, and `uv tool install` works the same way in PowerShell.
- **Qt runs on Wayland there, and that is fine.** WSLg advertises both a
  Wayland and an X11 display; Qt takes Wayland, which needs none of the
  `libxcb-*` libraries. Only if you set `QT_QPA_PLATFORM=xcb` yourself does
  the xcb list from {ref}`linux-system-libraries` become mandatory —
  `ethograph check` reads that variable and adjusts what it asks for.

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

**Audio embedded in a video (`.mp4`/`.mov`/`.avi`) is decoded to a WAV first,
never read in place.** libsndfile reads no video container, and a container's
AAC track has no sample-exact random access — which is exactly what the
waveform, spectrogram and playback clock ask for, one window at a time. So the
first time a video is used as an audio source, EthoGraph decodes its track once
(through PyAV, bundled with the `gui` extra — no separate ffmpeg install) into a
cached WAV under `~/.ethograph/audio_tracks/`, and every audio reader opens that
file instead. This happens wherever the audio path points at a video:

- **Drag & drop:** dropping a video that has an audio track offers **"extract
  audio"** — ticking it registers that track as an `audio_mic-N` stream, so the
  clip loads with a waveform and spectrogram. Untick it and the video loads with
  no audio at all.
- **Your own alignment / `.nwb`:** an audio stream may point straight at the
  `.mp4`. It is decoded on first use, with one log line naming the file, and
  reused from the cache on every later session.

The cache is keyed by source identity (path, size, mtime), so a re-recorded or
moved video is never served a stale extract. It is plain WAV — delete the folder
any time to reclaim the space; the next load simply decodes again.

Two reasons to still supply separate audio files when you can:

- **Sync.** AAC carries encoder priming/padding, so an extracted track can start
  or end a few milliseconds off the container's own timeline. For millisecond-
  accurate work, record or export the audio separately.
- **Disk and first-load time.** The extract is uncompressed and decoding a long
  video's track takes as long as decoding the audio. Converting in bulk up front
  is faster for a whole project:

  ```bash
  for f in *.mp4; do ffmpeg -i "$f" -vn -acodec pcm_s16le "${f%.mp4}.wav"; done
  ```

If a file cannot be decoded you get a single log line naming the file and the
reason (e.g. *"Cannot use the audio track of … the container has no readable
audio track"*) rather than a stream of errors.

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
