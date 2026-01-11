"""Path utilities with zero internal dependencies (stdlib only)."""
from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    """Find project root by walking up to find pyproject.toml or .git folder.

    Args:
        start: Starting path for search. Defaults to this file's location.

    Returns:
        Path to project root.

    Raises:
        FileNotFoundError: If no pyproject.toml or .git folder found in any parent.
    """
    path = (start or Path(__file__)).resolve()
    markers = ["pyproject.toml", ".git"]
    for parent in [path] + list(path.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    raise FileNotFoundError(
        f"Could not find project root (pyproject.toml or .git) starting from {path}"
    )


def gui_default_settings_path() -> Path:
    """Get the default path for gui_settings.yaml in the project root."""
    settings_path = get_project_root() / "gui_settings.yaml"
    settings_path.touch(exist_ok=True)
    return settings_path
