"""Generate executable Python code for temporal alignment setup."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jinja2

from ethograph.utils.validation import (
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    POSE_EXTENSIONS,
    EPHYS_EXTENSIONS,
)

if TYPE_CHECKING:
    from ethograph.gui.wizard_overview import ModalityConfig, WizardState
    from ethograph.gui.wizard_media_files import FilePattern

_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Template context dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModalityContext:
    name: str
    file_mode: str
    folder_path: str = ""
    single_file_path: str = ""
    nested_subfolders: bool = False
    extensions: list[str] = field(default_factory=list)
    regex: str | None = None
    has_trial_role: bool = False
    device_labels: list[str] | None = None
    constant_offset: float = 0.0


# ---------------------------------------------------------------------------
# Regex inference
# ---------------------------------------------------------------------------


def _infer_value_pattern(values: list[str]) -> str:
    if not values:
        return r'[^/._-]+'

    if all(v.isdigit() for v in values):
        lengths = {len(v) for v in values}
        return rf'\d{{{lengths.pop()}}}' if len(lengths) == 1 else r'\d+'

    m = [re.match(r'^([A-Za-z_]+)(\d+)$', v) for v in values]
    if all(m):
        prefixes = {x.group(1) for x in m}
        nums = [x.group(2) for x in m]
        if len(prefixes) == 1:
            prefix = re.escape(prefixes.pop())
            lengths = {len(n) for n in nums}
            if len(lengths) == 1:
                return rf'{prefix}\d{{{lengths.pop()}}}'
            return rf'{prefix}\d+'

    prefix = os.path.commonprefix(values)
    if prefix and len(prefix) > 1:
        return rf'{re.escape(prefix)}[^/._-]+'

    lengths = {len(v) for v in values if v}
    if len(lengths) == 1:
        return rf'[^/._-]{{{lengths.pop()}}}'

    if len(values) <= 10:
        return '|'.join(re.escape(v) for v in values)

    return r'[^/._-]+'


TOKEN_PATTERN = r'[^/\.]+'


def _segments_to_regex(pattern: FilePattern) -> str | None:
    from ethograph.gui.wizard_media_files import FOLDER_POSITION
    from ethograph.gui.wizard_media_files import _tokenize, _find_token_spans

    has_roles = any(
        seg.varying and seg.role and seg.role != "ignore"
        for seg in pattern.segments
        if seg.position != FOLDER_POSITION
    )
    if not has_roles or not pattern.files:
        return None

    ref_file = pattern.files[0].stem
    tokens = _tokenize(ref_file, pattern.tokenize_mode)
    spans = _find_token_spans(ref_file, tokens)

    regex_parts = []
    prev_end = 0

    for idx, seg in enumerate(pattern.segments):
        if seg.position == FOLDER_POSITION:
            continue
        if idx < len(spans) and prev_end < spans[idx][0]:
            delim = ref_file[prev_end:spans[idx][0]]
            regex_parts.append(re.escape(delim))
        if idx < len(spans):
            prev_end = spans[idx][1]

        if seg.varying and seg.role and seg.role != "ignore":
            value_pattern = _infer_value_pattern(seg.values)
            regex_parts.append(f"(?P<{seg.role}>{value_pattern})")
        elif not seg.varying:
            regex_parts.append(re.escape(seg.text))
        else:
            regex_parts.append(TOKEN_PATTERN)

    return "^" + "".join(regex_parts) + "$"


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _escape_path(path: str) -> str:
    return path.replace("\\", "\\\\")


def _escape_regex(pattern: str) -> str:
    return pattern.replace('"', '\\"')


def _get_extensions(name: str) -> list[str]:
    mapping = {
        "video": VIDEO_EXTENSIONS,
        "audio": AUDIO_EXTENSIONS,
        "pose": POSE_EXTENSIONS,
        "ephys": EPHYS_EXTENSIONS,
    }
    return sorted(mapping.get(name, []))


def _device_labels_for(name: str, state: WizardState) -> list[str] | None:
    if name in ("video", "pose") and state.camera_names:
        return state.camera_names
    if name == "audio" and state.mic_names:
        return state.mic_names
    return None


def _build_modality_context(name: str, cfg: ModalityConfig, state: WizardState) -> ModalityContext:
    if cfg.pattern and cfg.pattern.files:
        extensions = sorted({f.suffix.lower() for f in cfg.pattern.files if f.suffix})
    else:
        extensions = _get_extensions(name)[:3]

    regex = None
    has_trial_role = False
    if cfg.pattern and cfg.pattern.segments:
        raw = _segments_to_regex(cfg.pattern)
        if raw:
            regex = _escape_regex(raw)
            has_trial_role = any(seg.role == "trial" for seg in cfg.pattern.segments)

    return ModalityContext(
        name=name,
        file_mode=cfg.file_mode,
        folder_path=_escape_path(cfg.folder_path or ""),
        single_file_path=_escape_path(cfg.single_file_path or ""),
        nested_subfolders=cfg.nested_subfolders,
        extensions=extensions,
        regex=regex,
        has_trial_role=has_trial_role,
        device_labels=_device_labels_for(name, state),
        constant_offset=cfg.constant_offset,
    )


def _build_session_table_context(
    state: WizardState,
    modalities: list[ModalityContext],
) -> dict:
    has_trial_metadata = False
    trial_source = None
    for mod in modalities:
        if mod.has_trial_role:
            has_trial_metadata = True
            trial_source = mod.name
            break

    if state.trial_table_path:
        sep = "\\t" if state.trial_table_path.endswith(".tsv") else ","
        return {
            "session_table_source": "csv",
            "trial_table_path": _escape_path(state.trial_table_path),
            "trial_table_sep": sep,
        }

    if state.trial_table is not None and len(state.trial_table) <= 5:
        inline = {
            col: state.trial_table[col].tolist()
            for col in state.trial_table.columns
        }
        return {"session_table_source": "inline", "trial_table_inline": inline}

    if state.trial_table is not None:
        return {
            "session_table_source": "programmatic",
            "n_trials": len(state.trial_table),
            "has_start_stop": "start_time" in state.trial_table.columns,
        }

    if has_trial_metadata:
        return {
            "session_table_source": "metadata",
            "trial_metadata_source": trial_source,
        }

    file_source = next(
        (mod.name for mod in modalities if mod.name in ("video", "pose", "audio") and mod.folder_path),
        None,
    )
    if file_source:
        return {"session_table_source": "infer", "file_count_source": file_source}

    return {"session_table_source": "fallback"}


def _build_offsets(modalities: list[ModalityContext]) -> list[tuple[str, float]]:
    return [
        (mod.name, mod.constant_offset)
        for mod in modalities
        if mod.constant_offset != 0.0
    ]


def _build_template_context(state: WizardState) -> dict:
    modalities = [
        _build_modality_context(name, getattr(state, name), state)
        for name in ("video", "pose", "audio", "ephys")
        if getattr(state, name).enabled
    ]

    media_modalities = [m for m in modalities if m.name in ("video", "pose", "audio")]

    pose_mod = next((m for m in modalities if m.name == "pose"), None)
    pose_has_trial_metadata = pose_mod is not None and pose_mod.has_trial_role

    ctx: dict = {
        "modalities": modalities,
        "media_modalities": media_modalities,
        "pose_has_trial_metadata": pose_has_trial_metadata,
        "has_session_table": state.trial_table is not None,
        "offsets": _build_offsets(modalities),
        "output_path": _escape_path(state.output_path) if state.output_path else None,
    }
    ctx.update(_build_session_table_context(state, modalities))
    return ctx


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------


def _create_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=jinja2.StrictUndefined,
    )

    def format_glob(ext: str, prefix: str) -> str:
        return f'"{prefix}{ext}"'

    env.filters["format_glob"] = format_glob
    return env


_env: jinja2.Environment | None = None


def _get_env() -> jinja2.Environment:
    global _env
    if _env is None:
        _env = _create_jinja_env()
    return _env


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_alignment_code(state: WizardState) -> str:
    """Generate executable Python code that reproduces the user's alignment setup.

    Parameters
    ----------
    state
        Complete wizard state with modality configs and trial table.

    Returns
    -------
    Rendered Python code as a string.
    """
    ctx = _build_template_context(state)
    template = _get_env().get_template("alignment_code.py.j2")
    rendered = template.render(ctx)
    return _clean_blank_lines(rendered)


def _clean_blank_lines(text: str) -> str:
    """Collapse runs of 3+ blank lines into 2, strip trailing whitespace."""
    import re as _re

    text = _re.sub(r'\n{4,}', '\n\n\n', text)
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines)
