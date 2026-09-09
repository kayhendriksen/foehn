"""foehn — Download MeteoSwiss Open Government Data and convert to Parquet."""

__version__ = "0.4.0"

try:
    import polars as pl

    pl.DataFrame({"_": [0]})
except Exception as exc:
    raise ImportError(
        "Polars failed to load. On systems without AVX2 support (e.g. Databricks), "
        "install the compatible build:\n\n"
        '  pip install "foehn[databricks]"   # or: pip install polars-lts-cpu\n'
    ) from exc

# Imported for its side effect: binding ``foehn.client`` on the package, so the
# v0.4.0 spelling ``import foehn; foehn.client.load_last_run(...)`` resolves. A
# submodule is not an attribute of its package until something imports it, and
# v0.4.0's __init__ imported client for its own reasons — so the documented
# usage worked there and raised AttributeError here. The shim itself is
# deprecated; see foehn/client.py.
from foehn import client as client
from foehn.api import (
    download,
    inventory,
    list_datasets,
    load,
    metadata,
    open_dataset,
    parameters,
    stations,
    to_parquet,
    to_zarr,
)
from foehn.transfer import DownloadResult

__all__ = [
    "DownloadResult",
    "__version__",
    "client",
    "download",
    "inventory",
    "list_datasets",
    "load",
    "metadata",
    "open_dataset",
    "parameters",
    "stations",
    "to_parquet",
    "to_zarr",
]
