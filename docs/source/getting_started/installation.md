(target-installation)=
# Installation


### Install ffmpeg

ethograph uses [ffmpeg](https://ffmpeg.org/) for video and audio processing.
Check if it is already installed:

```bash
ffmpeg -version
```

If the command is not found, install it:

::::{tab-set}

:::{tab-item} Windows
1. Download `ffmpeg-release-essentials.zip` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2. Extract the zip and copy the `bin` folder to `C:\ffmpeg\bin`.
3. Add `C:\ffmpeg\bin` to your system **PATH**:
   - Search for *"Environment Variables"* in the Start menu.
   - Under *System variables*, select **Path** → **Edit** → **New** → paste `C:\ffmpeg\bin`.
4. Open a **new** terminal and verify: `ffmpeg -version`.
:::

:::{tab-item} macOS
```bash
brew install ffmpeg
```
:::

:::{tab-item} Linux
```bash
# Debian/Ubuntu
sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```
:::

::::

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

## Create a virtual environment

You can use either conda or uv to create the environment — conda is only
used for environment creation, not for installing ethograph itself.

::::{tab-set}

:::{tab-item} conda
```bash
conda create -y -n ethograph -c conda-forge python=3.12
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

The GUI bundles napari, PyQtGraph, audio support, and neural analysis tools:

```bash
uv pip install "ethograph[gui]"
```

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

### Everything

Install all optional dependencies (GUI + NWB):

```bash
uv pip install "ethograph[all]"
```

### Optional dependency groups

ethograph uses optional extras to keep the base install lightweight.
You can combine them as needed:

| Extra      | What it adds                                          |
|------------|-------------------------------------------------------|
| `gui`      | Full graphical interface  |
| `dandi`    | DANDI archive download client (heavy, opt-in)         |
| `dev`      | Testing and linting tools                             |
| `docs`     | Documentation build dependencies                     |

Combine extras with commas:

```bash
uv pip install "ethograph[gui,dandi, dev, docs]"
```

(install-from-source)=
## Install from source

Use this method while ethograph is not yet on PyPI, or whenever you
want to track the latest development version:

```bash
git clone https://github.com/Akseli-Ilmanen/ethograph.git
cd ethograph
uv pip install -e ".[gui]"
```

The `-e` flag installs in **editable mode** — code changes take effect
immediately without reinstalling.

## For developers

To install ethograph in editable mode with all optional dependencies and
development tools, see {doc}`../community/contributing`.


## Update the package

```bash
uv pip install -U "ethograph[gui]"
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
