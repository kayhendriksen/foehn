# Python API reference

Everything below is importable from the top-level package:

```python
import foehn
```

Tabular datasets come back as [Polars](https://pola.rs) DataFrames; gridded
datasets as [xarray](https://xarray.dev) Datasets. Every function takes the
dataset's short name — run `foehn.list_datasets()` or `foehn list` to see them.
For worked examples, see the [Python API guide](../python-api.md) and the
[gridded data guide](../grids.md).

## Discovering datasets

::: foehn.list_datasets

## Loading tabular data

::: foehn.load

## Metadata

::: foehn.parameters

::: foehn.stations

::: foehn.points

::: foehn.inventory

::: foehn.api.metadata

## Downloading and converting

::: foehn.download

::: foehn.DownloadResult
    options:
      # The fields are described in the class docstring; inspected one by one,
      # each would show int's own docstring instead.
      members: false

::: foehn.to_parquet

## Gridded data

::: foehn.open_dataset

::: foehn.to_zarr

## Diagnostics

::: foehn.show_versions
