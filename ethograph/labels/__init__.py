"""Label representations: interval-based (GUI/storage) and dense (ML pipelines)."""

import logging

logger = logging.getLogger(__name__)

# Register ethograph-seq Crowsetta format on import
try:
    import ethograph.labels.crowsetta_format  # noqa: F401
except Exception as exc:  # pragma: no cover - optional dependency registration
    logger.warning("Skipping crowsetta format auto-registration: %s", exc)
