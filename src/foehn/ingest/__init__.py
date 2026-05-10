"""Delta ingestion for foehn datasets.

Public entry point: :func:`ingest`. The pipeline primitives
:func:`foehn.ingest.pipeline.run_tabular` and
:func:`foehn.ingest.pipeline.run_radar` are also available for callers
that already hold their own ``SparkSession`` and adapters.
"""

from foehn.ingest.entry import ingest

__all__ = ["ingest"]
