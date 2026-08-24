(target-installation)=
# Installation


### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.
ethograph uses uv for installation.

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::

:::{tab-item} Windows
```
winget install astral-sh.uv
```
Works from both PowerShell and Command Prompt. `winget` is built into Windows 11.
:::

::::

## Quick install

Use this if you just want to **run the ethograph GUI** — for teaching, for
annotating data, or to try it out. It is one command, and there is no
environment to create or activate.

```bash
uv tool install --python 3.12 "ethograph[gui,audio]"
```

Then launch it from any terminal:

```bash
ethograph launch
```

`uv tool install` puts ethograph in its own isolated environment and adds the
`ethograph` command to your PATH, so it never clashes with your other Python
projects and is always available without activating anything.

```{important}
On a fresh machine the first `ethograph launch` may fail with *"'ethograph' is
not recognized as the name of a cmdlet..."* (Windows) or *"command not found"*
(macOS/Linux). uv has not yet added its bin directory to your `PATH`. Fix it
once with:

    uv tool update-shell

then **close the terminal and open a new one** — `PATH` is only read when a
shell starts, so the window you ran it in will keep failing. See
{ref}`command-not-found` for shortcut-based alternatives.
```

```{tip}
To update later, run `uv tool upgrade ethograph`; to remove it,
`uv tool uninstall ethograph`.
```

## Create a virtual environment

Use this approach instead if you want to **write scripts or code against
ethograph** — import `TrialTree`, build pipelines, or develop the package. Here
ethograph is installed *into an environment you activate*, alongside whatever
else you import, rather than as a standalone tool.

You can use either conda or uv to create the environment — conda is only
used for environment creation, not for installing ethograph itself.

::::{tab-set}

:::{tab-item} conda
```bash
conda create -y -n ethograph python=3.12
conda activate ethograph
```
:::

:::{tab-item} uv
```bash
uv venv --python=3.12
```

Activate it:
```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```
:::

::::

## Install the package

```{important}
Make sure your virtual environment is activated before running any install
commands. You should see the environment name (e.g. `(ethograph)`) in your
terminal prompt.
```

### With the GUI (recommended)

The GUI bundles PyQtGraph, pygfx (via pynaviz), and neural analysis tools:

```bash
uv pip install "ethograph[gui]"
```

#### Adding audio support

Audio support (waveform display, spectrogram, vocalisation analysis) is an optional extra.

::::{tab-set}

:::{tab-item} macOS / Windows
```bash
uv pip install "ethograph[gui,audio]"
```
:::

:::{tab-item} Linux
First install the PortAudio system library **and the ALSA plugins**, then
install the audio extra:
```bash
sudo apt install libportaudio2 libasound2-plugins   # Debian / Ubuntu
uv pip install "ethograph[gui,audio]"
```

`libasound2-plugins` is the ALSA→PipeWire/PulseAudio bridge; without it
PortAudio can't reach a modern sound server. See {ref}`no-audio-device` if
playback is still silent.
:::

::::

If you are using a **conda environment**, optionally create a desktop shortcut:

```bash
ethograph shortcut
```

```{admonition} Launching the GUI
:class: tip

After installation, launch ethograph from the terminal:

    ethograph launch

Or use the desktop/Start Menu shortcut created above.
```

Global settings live in `~/.ethograph` (override with the `ETHOGRAPH_HOME` environment variable).

### Core only (library)

The core package includes the `TrialTree` data structure, xarray utilities,
feature extraction, and label I/O — no GUI, no audio, no NWB. This is useful
when you want to use `TrialTree` as a data structure in your own scripts or
pipelines without pulling in GUI dependencies.

```bash
uv pip install ethograph
```


### DANDI archive downloads

To browse and download datasets from the [DANDI
archive](https://dandiarchive.org/) via the GUI wizard:

```bash
uv pip install "ethograph[dandi]"
```

### Faster scrubbing in long videos (optional)

EthoGraph works **fully without ffmpeg** — every feature is available, including
video-motion extraction, which runs in-process through PyAV (bundled with the
`gui` extra). The only thing ffmpeg adds is *proxy generation*: a low-resolution
copy of a video that makes seeking through long, high-resolution recordings
smoother. Video always plays at full resolution without it, so for short teaching
clips it buys nothing.

If you do work with long recordings, add ffmpeg with:

```bash
uv pip install "ethograph[proxy]"
```

This bundles a private ffmpeg binary (via `imageio-ffmpeg`) with no PATH setup.
If you already have ffmpeg installed on your system, EthoGraph picks it up
automatically — or point it at a specific executable with the `ETHOGRAPH_FFMPEG`
environment variable.

```{note}
The bundled ffmpeg does **not** include NVENC, so GPU (`cuda`) proxy encoding
falls back to software `libx264`. For NVENC, install ffmpeg from conda-forge
(`conda install -c conda-forge ffmpeg`) or your system package manager.
```

### Optional dependency groups

ethograph uses optional extras to keep the base install lightweight.
You can combine them as needed:

| Extra          | What it adds                                                        |
|----------------|---------------------------------------------------------------------|
| `gui`          | Full graphical interface (PyQtGraph, pygfx/pynaviz, neural tools)   |
| `audio`        | Waveform, spectrogram, vocalisation analysis (`sounddevice` etc.)   |
| `dandi`        | DANDI archive download client (heavy, opt-in)                       |
| `proxy`        | Bundled ffmpeg for faster scrubbing in long videos (optional)       |
| `dev`          | Testing and linting tools                                            |
| `docs`         | Documentation build dependencies                                    |

```{note}
Linux users adding `audio` must first install the PortAudio system library
**and the ALSA plugins**:
`sudo apt install libportaudio2 libasound2-plugins`
```

Combine extras with commas:

```bash
uv pip install "ethograph[gui,audio,dandi,dev,docs]"
```


## For developers

To install latest development version in editable mode see {doc}`../community/contributing`.


## Update the package

```bash
uv pip install -U "ethograph[gui]"
# With audio:
uv pip install -U "ethograph[gui,audio]"
```

```{hint}
If the update doesn't seem to work, try creating a fresh environment
and reinstalling from scratch.
```

(target-keypoint-fill)=
## Keypoint labelling backends (optional)

**Tools ▸ Keypoint labelling…** lets you label a few frames by clicking the
video and fill the rest automatically.

| Backend | Method | Uses video | Speed | Hardware |
|---------|--------|------------|-------|----------|
| **Spline** (default) | Monotone piecewise cubic (PCHIP) interpolation per keypoint, over that keypoint's own labelled frames[^pchip] | No — geometry only | Instant; nothing is decoded | CPU |
| **Optical flow** | Pyramidal Lucas-Kanade sparse tracking, run forward and backward across each gap[^lk] | Yes | Roughly real-time | CPU |
| **PosePAL (CoTracker3 + refinement)** | A transformer point tracker[^cotracker] whose per-keypoint appearance features are first fitted to the frames you labelled[^posepal], then run forward and backward across each gap | Yes | A few minutes for the fit, once; seconds per fill after that | **GPU only** — CUDA or Apple Silicon |

Spline and optical flow need **nothing extra**, and neither does the **Detect**
tab, which reads AprilTag `tag36h11` markers and feeds them to whichever backend
you fill with — OpenCV and `pupil-apriltags` both come with `ethograph[gui]`.

Only PosePAL is a separate install — GPU only, and `--torch-backend=auto`
matters since PyPI's Windows torch wheels are CPU-only:

```bash
uv pip install --torch-backend=auto torch "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
```

```{seealso}
This page covers *installing* the backends. Which one to reach for, how the
labelling loop works, what the PosePAL fit costs and when to train a DeepLabCut
detector instead are covered in {doc}`../advanced/keypoint_labelling/index`.
```

## Segmentation pipeline (model training)

The {doc}`segmentation pipeline <../advanced/segment/index>` learns your
curated state labels and predicts them back into the GUI. It is scripted,
not a command line: one config becomes a `Project` with a method per stage.
It needs PyTorch; install a build matching your GPU first, then the extra:

```bash
uv pip install --torch-backend=auto torch torchvision
uv pip install "ethograph[model]"
```

```python
import ethograph as eto

eto.segment.architectures()         # lists the available models
```

`--torch-backend=auto` matters on Windows, where PyPI's default torch wheels
are CPU-only. Training runs on CPU too, just slowly.

```{tip}
Using `conda-forge` (`conda create -y -n ethograph -c conda-forge python=3.12`)
can help here: it keeps all conda-installed packages on one channel, avoiding
ABI conflicts between `defaults` and `conda-forge` builds of shared libraries
that PyTorch/CUDA packages depend on.
```

[^pchip]: Fritsch, F. N. & Carlson, R. E. (1980). [Monotone Piecewise Cubic Interpolation](https://doi.org/10.1137/0717021). *SIAM Journal on Numerical Analysis*, 17(2), 238–246. Implemented by [`scipy.interpolate.PchipInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

[^lk]: Lucas, B. D. & Kanade, T. (1981). [An Iterative Image Registration Technique with an Application to Stereo Vision](https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf). *IJCAI*, 674–679. The pyramidal form used here is Bouguet, J.-Y. (2001), [Pyramidal Implementation of the Lucas Kanade Feature Tracker](https://robots.stanford.edu/cs223b04/algo_tracking.pdf), via [`cv2.calcOpticalFlowPyrLK`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323).

[^cotracker]: Karaev, N., Makarov, I., Wang, J., Neverova, N., Vedaldi, A. & Rupprecht, C. (2024). [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831). [Project page](https://cotracker3.github.io/) · [GitHub](https://github.com/facebookresearch/co-tracker)

[^posepal]: Pan, Z., Pan, B., Yang, G., Harley, A. W. & Guibas, L. (2025). [Animal Pose Labeling Using General-Purpose Point Trackers](https://arxiv.org/abs/2506.03868). Reference implementation: [PosePAL](https://github.com/Zhuoyang-Pan/PosePAL). EthoGraph implements the method against upstream CoTracker3 rather than the authors' fork; the optimiser settings follow the paper (Adam, 1e-3 → 1e-5, Huber tracking loss, L1 pull-back weighted 0.01).
