"""foehn — Download MeteoSwiss Open Government Data and convert to Parquet."""

__version__ = "0.2.5"

from foehn.api import convert, download, list_datasets, load

__all__ = ["__version__", "convert", "download", "list_datasets", "load"]
