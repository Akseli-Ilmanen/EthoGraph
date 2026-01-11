# Installation

## Requirements

- Python >= 3.8
- napari

## Install from source

```bash
git clone https://github.com/yourusername/ethograph.git
cd ethograph
pip install -e .
```

## Install with all dependencies

For napari and Qt dependencies:

```bash
pip install -e ".[all]"
```

## Verify installation

```python
from ethograph import __version__
print(__version__)
```

## Launch the GUI

After installation, you can launch EthoGraph in two ways:

### From command line

```bash
ethograph-gui
```

### As napari plugin

1. Launch napari
2. Go to `Plugins > ethograph-gui`
