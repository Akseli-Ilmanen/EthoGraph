# Handoff: audio playback timing + video render freeze

Written 2026-08-18 after triaging the Windows tester reports in this folder
(`ethograph_annotation_feasibility_report.md`,
`ethograph_windows_author_overview_2026-08-18.md`). Those reports test
`ethograph 0.2.12`; this repo is ahead of that release, and every anchor below
was verified against the current tree, not against 0.2.12.

Several fixes from the same triage are **already landed** — don't redo them:

- `gui/widgets_labels.py` + `gui/widgets_data.py`: `Qt.CheckState(qt_state) == Qt.Checked`
  (the branch-overlay hide/re-show bug the tester patched locally).
- `gui/app_state.py` `get_audio_source()`: an unknown `audio_source_map` key now
  returns `(None, 0)` instead of being read as a file name.
- `gui/pygfx_video.py` `_detach_canvas()`: `CameraView.clear()` tolerates a
  canvas whose C++ side is already gone.
- `gui/label_drawing_mixin.py` `show_pending_label()` / `clear_pending_label()`:
  a state label being drawn now shows a dashed anchor on its onset plus a faint
  cursor-tracking preview on every panel. Covered by
  `tests/test_unit/test_pending_label_preview.py`.
- `gui/widgets_labels.py` `_update_branch_header_styles()`: the editable branch
  says "✎ … — editing" rather than differing only by shade, and the visibility
  checkbox's tooltip says it does not choose what you edit.

What is left are the two hard clusters. They are independent — do them in
either order, but keep them in separate commits.

---

## Work item A — audio playback timing

Three symptoms from the reports, all in the same code path:

1. **Play → Stop bursts shorter than ~0.4 s do not commit.** Audio is audible,
   but the video frame and `current_frame` never move, and the next Play
   restarts from where the previous burst began.
2. **Noticeable delay on every Play/Stop.** Described as "workable, clearly
   better than the old version", so it is a polish item, not a blocker.
3. **Changing the audio channel during playback** updates the audio trace and
   spectrogram, but the *audible* channel only changes after Stop and the next
   Play.

### A1 — the burst bug (root cause is exact)

`gui/audio_clock.py:143-152`:

```python
real_s = self._idx / self._out_rate - self._latency_s
return max(0.0, min(real_s * self._speed, self.duration_s))
```

`_latency_s` is the device's output latency (`gui/audio_clock.py:112`,
`stream.latency`) — typically 0.1–0.3 s on Windows MME/WASAPI. For the first
`_latency_s` of playback the subtraction goes negative and `max(0.0, …)` pins
the result at exactly `0.0`. `VideoSync._advance_from_clock`
(`gui/video_sync.py:375`) then computes `frame == _clock_start_frame` on every
tick, so `_apply_frame` never advances `current_frame`. The reported ~0.4 s
threshold is that latency window.

Subtracting the latency is right in principle — `_idx` counts frames *handed to*
the device, which runs ahead of what is audible. The bug is the clamp, which
turns "I don't know yet" into "nothing has played".

The fix needs a position that is monotonic from t=0. Two workable approaches;
pick one and say why in the commit message:

- Keep a `time.perf_counter()` reference at stream start and report
  `min(wall_elapsed, idx / out_rate)` while inside the latency window, handing
  over to the device counter once it overtakes.
- Use `time_info.outputBufferDacTime` / `currentTime` from `_callback` to derive
  a true audible position instead of inferring one from `_idx`.

Do not simply drop the latency subtraction — that trades a stuck marker for a
marker that leads the sound by the full device latency, which is the drift the
`AudioClock` exists to prevent.

`VideoSync.stop()` (`gui/video_sync.py:211`) should also commit the clock's
final position to `current_frame` before tearing the clock down, so a burst that
did play leaves the playhead where the listener last heard it.

### A2 — Play/Stop latency

`gui/video_sync.py:189` builds the clock over
`[current_frame / fps, total_frames / fps]` — **everything to the end of the
trial** — and `AudioClock._prepare_output` (`gui/audio_clock.py:67-88`) then runs
`resample_poly(media, 48000, int(fs * speed))` on all of it, synchronously, on
the GUI thread, on every Play press.

For the tester's 24414 Hz files that ratio reduces to `up=8000, down=4069`,
which designs a ~160k-tap FIR and runs it over ~3M samples. The cost scales with
how early in the trial you press Play.

Two changes:

- Bound the resampling ratio (e.g. `Fraction(OUTPUT_RATE / (fs * speed)).limit_denominator(...)`),
  or resample to the device's own preferred rate instead of a fixed 48 kHz. Keep
  the existing SciPy-absent fallback path working.
- Preload a bounded look-ahead span with refill rather than the whole remaining
  trial. This is the larger change and it interacts with A1 — the position
  bookkeeping has to survive a refill boundary.

`OUTPUT_RATE = 48000` exists so that any source rate and any playback speed can
play (see the module docstring — ultrasonic recordings review at slow
time-expansion speeds). Whatever replaces it must keep that property.

### A3 — channel switching mid-playback

`_build_audio_clock` (`gui/video_sync.py:389`) is called once, from `start()`.
`playback_mic_key` changes during playback therefore cannot reach the open
stream. Rebuild the clock from the current elapsed position when the mic or
channel changes while `is_playing`. `app_state.playback_mic_selection()` already
filters stale keys, and `get_audio_source` now refuses unknown ones, so the
resolution side is sound — only the rebuild trigger is missing.

### Verifying A

Needs a real audio device, so it has to be checked on Windows by hand. The
tester's pass criteria:

- A 0.2 s Play → Stop burst advances the playhead and the next Play continues
  from there.
- Audio, video frame and playhead stay synchronized after zooming and after
  repeated stop/start.
- Switching channel during playback changes what you hear without a Stop.

Add unit coverage for the clock arithmetic itself (no device): construct an
`AudioClock`, drive `_idx` by hand, and assert `elapsed_s()` is monotonic and
non-zero inside the latency window. `tests/test_unit/test_video_sync_ready.py`
is the nearest existing pattern.

---

## Work item B — video freeze while audio and playhead continue

The tester reproduced this at least three times. The strongest capture is
`.home/ethograph_video_debug_freeze_20260818_152425.log` on their machine —
worth asking for it if you want the raw evidence.

### What the evidence says

At freeze time: `VideoSync._advance_from_clock`, `CameraView.seek_video_frame`
and `PlotVideo.buffer_frame` all still running, `audio_clock=True`, worker and
buffer threads alive, `pending_ui_q=1`, `needs_redraw=True`, and
**`PlotVideo.animate()` had not run for ~1000 s**. Audio and the timeline kept
moving; only the image was stale.

That maps onto pynaviz's design. `PlotVideo.animate()`
(`site-packages/pynaviz/audiovideo/video_plot.py:553-588`) ends with
`self.canvas.request_draw(self.animate)` — **the loop continues only because its
last line re-arms it** — and its `try` catches nothing but `queue.Empty`. Any
raise from `_set_time_text()`, `texture.update_full()` or
`renderer_request_draw()`, or one paint dropped by the rendercanvas Qt scheduler
(hidden dock, resize, swallowed present error), ends the chain permanently.
Nothing in ethograph notices.

Two independent observations corroborate it: switching playback mode does not
recover (mode changes never touch the canvas), and **closing the frozen video
panel and adding a new one does** (a new canvas gets a fresh chain).

### What to build

1. **A resilient animate wrapper.** After `PlotVideo` is created in
   `CameraView.set_video` (`gui/pygfx_video.py`), replace `plot.animate` on the
   *instance* with a wrapper that calls the original inside a `try`, stamps a
   heartbeat timestamp, and re-arms `canvas.request_draw` even when the call
   raised. Because the original's last line re-arms with `self.animate`, an
   instance attribute resolves back to the wrapper — the chain stays wrapped.
2. **A watchdog.** A `QTimer` (in `VideoSync` or `CameraView`) that, while
   playback is live, checks the heartbeat age; past ~1 s it re-arms
   `canvas.request_draw(plot.animate)` and logs **once** per stall. This is the
   "stale-render detection plus automatic reset" the report asks for.
3. **A manual escape hatch.** A Tools ▸ Reset video view command that rebuilds
   the primary `PlotVideo` in place — the recovery already proven to work, made
   reachable without closing and re-adding the panel.

If the wrapper proves the exception theory, the fix belongs upstream in pynaviz
too; ethograph should keep the watchdog either way.

### Traps

- `_disarm_present` (`gui/pygfx_video.py`) permanently neutralises
  `_rc_request_paint` on a canvas. It is only reached from teardown today, but a
  disarmed canvas that got reused would produce *exactly* this freeze signature.
  Have the wrapper assert the canvas is not disarmed before re-arming.
- Recreating a `PlotVideo` spawns a worker that re-imports `av`/`pygfx`/
  `pynapple` (~2 s on Windows) while `close()` waits only `join(timeout=2)`.
  A close-then-create cycle kills the new worker with
  `FileNotFoundError: … 'wnsm_…'` — the tester saw exactly that warning near one
  freeze. See the "Video sync" section of `CLAUDE.md`; the reset command must not
  reintroduce that race.
- Never `shell.removeDockWidget` a dock holding GL content (hide + `deleteLater`),
  and never re-apply a `QMainWindow.restoreState` blob to re-place video docks.
  Both are documented native-crash paths.

### Verifying B

The freeze is intermittent and Windows-only, so instrument first and fix second.
Get the watchdog logging a stall before trusting any fix, and keep the tester's
`proxy1280_audio3ch` multi-panel session (two spectrograms on different
channels, a second audio trace) as the reproduction recipe — that is the
configuration that produced the strongest capture.

---

## Related panel-lifecycle bug (medium, not in either cluster)

Same reports, different cause, worth doing whenever someone is next in
`video_manager.py`:

`RuntimeError: wrapped C/C++ object of type QRenderCanvas has been deleted` via
`on_trial_changed → update_video → _cleanup_primary_video → primary_view.clear()`.
The landed `_detach_canvas` guard stops it aborting the trial change, but the
underlying lifecycle is still wrong:

- The shell's `VideoDock` has **no close handling at all**. Extras get
  `VideoArea.eventFilter → remove_extra()`; the primary gets nothing, so closing
  it merely hides the dock while the view keeps a live plot, worker and canvas.
- `MetaWidget._add_camera_view` branches on `vm.primary_view.has_video`, not on
  whether the dock is visible — so re-adding "Video (cam)" after closing the
  video dock forks an *extra* view while an invisible primary lingers.

Fix: give the VideoDock the same close → teardown path the extras have, and make
`_add_camera_view` re-show a hidden primary instead of creating an extra.

---

## Out of scope

- **Embedded MP4/AAC audio.** Already surfaces the correct message, and the
  drop-time "extract audio" checkbox is the sanctioned path. Documentation, not
  code.
- **Mouse-wheel panning instead of zooming.** `Lock Axes` and `Nav → Fixed
  window` behaving as designed. A tooltip or status hint at most.
