(target-installation)=
# Installation


### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.
ethograph uses uv for installation regardless of how you create your
virtual environment.

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

| Extra      | What it adds                                                        |
|------------|---------------------------------------------------------------------|
| `gui`      | Full graphical interface (PyQtGraph, pygfx/pynaviz, neural tools)   |
| `audio`    | Waveform, spectrogram, vocalisation analysis (`sounddevice` etc.)   |
| `dandi`    | DANDI archive download client (heavy, opt-in)                       |
| `proxy`    | Bundled ffmpeg for faster scrubbing in long videos (optional)       |
| `dev`      | Testing and linting tools                                           |
| `docs`     | Documentation build dependencies                                    |

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

## Model training (experimental)

```{warning}
Model training is **in development and not well documented**.
```

Install PyTorch with the correct CUDA version for your system, then install
the model extra along with `omegaconf`:

```bash
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
uv pip install "ethograph[model]" omegaconf
```

```{tip}
Using `conda-forge` (`conda create -y -n ethograph -c conda-forge python=3.12`)
can help here: it keeps all conda-installed packages on one channel, avoiding
ABI conflicts between `defaults` and `conda-forge` builds of shared libraries
that PyTorch/CUDA packages depend on.
```
