"""foehn — Download MeteoSwiss Open Government Data and convert to Parquet."""

__version__ = "0.2.12"

try:
    import polars as pl

    pl.DataFrame({"_": [0]})
except Exception as exc:
    raise ImportError(
        "Polars failed to load. On systems without AVX2 support (e.g. Databricks), "
        "install the compatible build:\n\n"
        '  pip install "foehn[databricks]"   # or: pip install polars-lts-cpu\n'
    ) from exc

from foehn.api import download, inventory, list_datasets, load, parameters, stations, to_parquet

__all__ = ["__version__", "download", "inventory", "list_datasets", "load", "parameters", "stations", "to_parquet"]
