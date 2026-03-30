"""Label representations: interval-based (GUI/storage) and dense (ML pipelines)."""

# Register ethograph-seq Crowsetta format on import
try:
    import ethograph.labels.crowsetta_format  # noqa: F401
except ImportError:
    pass
