from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ethograph")
except PackageNotFoundError:
    # package is not installed
    pass
