"""foehn — Download MeteoSwiss Open Government Data and convert to Parquet."""

__version__ = "0.1.1"

from foehn.api import convert, download, list_collections
from foehn.collections import discover

__all__ = ["__version__", "convert", "discover", "download", "list_collections"]
