"""foehn — Download MeteoSwiss Open Government Data and convert to Parquet."""

__version__ = "0.2.0"

from foehn.api import convert, fetch, list_collections
from foehn.collections import discover

__all__ = ["__version__", "convert", "discover", "fetch", "list_collections"]
