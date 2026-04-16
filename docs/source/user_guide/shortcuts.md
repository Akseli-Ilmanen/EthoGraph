(target-shortcuts)=
# Keyboard Shortcuts


## Video/Audio Control

| Shortcut | Action |
|----------|--------|
| `Space` | Toggle play/pause video and audio (or audio-only in no-video mode) |
| `v` | Play selected label segment |
| `Left` / `Right` | Step one frame backward / forward (video mode) or one time-step (no-video mode) |
| `Shift+Left` / `Shift+Right` | Jump backward / forward by customizable time step (see "Jump step (ms)" in Navigation) |

## Navigation

| Shortcut | Action |
|----------|--------|
| `Up` | Previous trial |
| `Down` | Next trial |
| `Ctrl+Up` / `Ctrl+Down` | Previous / next channel (ephys or audio mic) |

## Mouse Controls

| Action | Function |
|--------|----------|
| **Left Click** | Select label (when not in label mode) |
| **Double Left Click** | Autoscale axes |
| **Right Click** | Seek video to clicked time |
| **Mouse wheel** | Zoom in/out both axes |
| **Right Click + drag horizontally** | Zoom in/out along the time axis |
| **Right Click + drag vertically** | Zoom in/out along the y-axis |

## Labelling

| Shortcut / Action | Description |
|-------------------|-------------|
| `1-9`, `0`, `Q-P`, `A-L` | Activate label |
| Click twice on line plot | Define label boundaries (set start/end) |
| Left-click on label | Select existing label |
| `Ctrl+E` | Edit selected label boundaries (after selecting label, click twice for new boundaries) |
| `Ctrl+D` | Delete selected label (after selecting label) |
| `Ctrl+S` | Save `labels.tsv` file.
| `Ctrl+Y` | Switch between labels and predictions |
| `Ctrl+V` | 'Verify predictions' by editing once or this shortcut |

## Selection Cycling

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Cycle to next feature |
| `Ctrl+I` | Cycle to next individual |
| `Ctrl+K` | Cycle to next keypoint |
| `Ctrl+C` | Cycle to next camera |
| `Ctrl+M` | Toggle to last-visited microphone, or cycle to next |

## Plot Controls

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Refresh line plot |
| `Ctrl+A` | Toggle autoscale |
| `Ctrl+L` | Toggle lock axes |
| `Ctrl+Enter` | Apply current plot settings |

## Panel Toggles

| Shortcut | Action |
|----------|--------|
| `Shift+A` | Toggle audio trace panel |
| `Shift+S` | Toggle spectrogram panel |
| `Shift+E` | Toggle ephys panel |
| `Shift+F` | Toggle feature plot panel |

## Changepoint Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+Right` | Jump to next changepoint (audio CPs if audio/spectrogram panel was last clicked, otherwise kinematic CPs) |
| `Ctrl+Left` | Jump to previous changepoint (same panel context) |

## Ephys Trace

| Shortcut | Action |
|----------|--------|
| `Ctrl+H` | Cycle neural view: 1-ch Trace -> Multi Trace -> Raster |
| **Ctrl+Wheel** | Adjust display gain |

## 3D Space

| Control | Action |
|---------|--------|
| **Left drag** | Orbit (rotate around center) |
| **Middle drag** | Pan (move look-at point) |
| **Scroll** | Zoom in / out |
| **Arrow keys** | Fine rotation (azimuth / elevation) |
