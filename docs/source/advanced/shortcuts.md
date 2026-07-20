(target-shortcuts)=
# Keyboard Shortcuts

Plain-letter shortcuts (`Space`, `V`, label keys, arrow keys) are suppressed
while you are typing in a text field, spin box or editable combo box.

## Video/Audio Control

| Shortcut | Action |
|----------|--------|
| `Space` | Toggle play/pause video and audio (or audio-only in no-video mode) |
| `V` | Play selected label segment |
| `Left` / `Right` | Step one frame backward / forward (video mode) or one time-step (no-video mode) |
| `Shift+Left` / `Shift+Right` | Jump backward / forward by customizable time step (see "Jump step (ms)" in Navigation) |
| `Ctrl+Space` | Stop screen recording |

## Navigation

| Shortcut | Action |
|----------|--------|
| `Up` | Previous trial |
| `Down` | Next trial |
| `Ctrl+Up` / `Ctrl+Down` | Previous / next channel (ephys or audio mic) |
| `Ctrl+P` | Cycle the sync mode combo |

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
| `1-9`, `0` | Activate label 1-10 |
| `Q W E R T Z U I O P` | Activate label 11-20 |
| `A S D F G H J K L` | Activate label 21-29 |
| Click twice on line plot | Define label boundaries (set start/end) |
| Left-click on label | Select existing label |
| `Ctrl+E` | Edit selected label boundaries (after selecting label, click twice for new boundaries) |
| `Ctrl+D` | Delete selected label (after selecting label) |
| `Ctrl+S` | Save `labels.tsv` file |
| `Ctrl+Y` | Switch between labels and predictions |
| `Ctrl+V` | 'Verify predictions' by editing once or this shortcut |
| `Shift+B` | Switch the Main labels slot to the previously-selected branch |

## Selection Cycling

| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Toggle to last-visited feature, or cycle to next |
| `Ctrl+I` | Toggle to last-visited individual, or cycle to next |
| `Ctrl+K` | Toggle to last-visited keypoint, or cycle to next |
| `Ctrl+C` | Cycle to next camera |
| `Ctrl+M` | Toggle to last-visited microphone, or cycle to next |
| `Shift+K` | Toggle the active space plot's keypoint dim to its previous value |

## Plot Controls

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | Open the "Add panel" popup |
| `Ctrl+R` | Refresh line plot |
| `Ctrl+A` | Toggle autoscale |
| Left double-click | Autoscale (once) |
| `Ctrl+L` | Toggle lock axes |
| `Ctrl+Enter` | Apply current plot settings |
| `Ctrl+G` | Cycle the feature view mode combo |

## Layout

| Shortcut | Action |
|----------|--------|
| `Ctrl+0` | Show / hide the control sidebar |
| `Ctrl+Z` | Zen mode — hide the sidebar and skip its refreshes |

## Changepoint Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+Right` | Jump to next changepoint (audio CPs if audio/spectrogram panel was last clicked, otherwise kinematic CPs) |
| `Ctrl+Left` | Jump to previous changepoint (same panel context) |
| `Ctrl+B` | Toggle changepoint correction |

## Ephys Trace

| Shortcut | Action |
|----------|--------|
| `Ctrl+H` | Cycle neural view: Multi Trace -> Raster |
| `Ctrl+=` / `Ctrl+-` | Increase / decrease channel spacing |
| `Alt+Right` / `Alt+Left` | Jump to next / previous spike |
| **Ctrl+Wheel** | Adjust display gain |

## 3D Space

| Control | Action |
|---------|--------|
| **Left drag** | Orbit (rotate around center) |
| **Middle drag** | Pan (move look-at point) |
| **Scroll** | Zoom in / out |
| **Arrow keys** | Fine rotation (azimuth / elevation) |
