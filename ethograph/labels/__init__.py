"""Label representations: interval-based (GUI/storage) and dense (ML pipelines)."""

import logging

from ethograph.utils.logging import enable_console_logging

logger = logging.getLogger(__name__)

# Register ethograph-seq Crowsetta format on import
try:
    import ethograph.labels.crowsetta_format  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency registration
    logger.warning("Skipping crowsetta format auto-registration: %s", exc)

# Onset-model training prints its progress (see labels/onset_model.py) — no
# reason to hide it by default.
enable_console_logging(logger_name="ethograph.labels.onset_model")
