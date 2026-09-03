"""MeteoSwiss's ODIM-H5 radar composites, and what it takes to read one.

The radar half of what :mod:`foehn.meteocsv` is to the CSV path: upstream's file
conventions, below the reader that uses them. The OGD radar products are ODIM
``COMP`` images rather than polar volumes, so xradar does not apply and there is
no xarray engine for them — the whole format is this module.

Takes ``xr`` as an argument rather than importing it: xarray is an optional
dependency, and :mod:`foehn.grids` already owns the one guard that reports a
missing 'grids' extra. Nothing else here is optional to foehn, and nothing here
touches the network — one file in, one Dataset out.
"""

from __future__ import annotations

import contextlib
from pathlib import Path


def _attr(group, key, default=None):
    """Read an HDF5 attribute, decoding bytes to str."""
    val = group.attrs.get(key, default)
    return val.decode() if isinstance(val, bytes) else val


def open_composite(xr, path: Path, *, data: bytes | None = None):
    """Read a MeteoSwiss ODIM-H5 Cartesian radar composite into an xarray Dataset.

    The OGD radar products (CombiPrecip precipitation, hail) are ODIM ``COMP``
    images, not polar volumes — a single 2-D grid at ``/dataset1/data1/data`` on
    the Swiss projection. We apply the ODIM ``gain``/``offset`` scaling, map the
    ``nodata`` sentinel (outside radar coverage) to NaN and ``undetect`` (nothing
    detected) to 0, and turn the ``/where`` projection metadata into LV95 x/y
    coordinates via pyproj so the result lines up with the NetCDF Swiss grids.
    """
    import io

    import h5py
    import numpy as np
    import pyproj

    # ``data`` decodes one revision the caller already holds, rather than
    # whatever is at *path* by the time we get there. ``path`` is still what the
    # messages name.
    source = io.BytesIO(data) if data is not None else path
    with h5py.File(source, "r") as f:
        obj = _attr(f["what"], "object", "")
        if obj != "COMP":
            raise ValueError(f"{path.name}: expected an ODIM 'COMP' composite, got object={obj!r}.")

        node = f["dataset1"]["data1"]
        dwhat = node["what"]
        gain = float(_attr(dwhat, "gain", 1.0))
        offset = float(_attr(dwhat, "offset", 0.0))
        nodata = float(_attr(dwhat, "nodata", np.nan))
        undetect = float(_attr(dwhat, "undetect", np.nan))
        quantity = str(_attr(dwhat, "quantity", "data"))
        raw = node["data"][:].astype("float64")

        where = f["where"]
        xsize, ysize = int(where.attrs["xsize"]), int(where.attrs["ysize"])
        xscale, yscale = float(where.attrs["xscale"]), float(where.attrs["yscale"])
        projdef = _attr(where, "projdef")
        ul_lon, ul_lat = float(where.attrs["UL_lon"]), float(where.attrs["UL_lat"])

        what = f["what"]
        date, time = _attr(what, "date", ""), _attr(what, "time", "")
        long_name = ""
        if "how" in f and "MeteoSwiss" in f["how"]:
            long_name = _attr(f["how"]["MeteoSwiss"], "long_name", "")

    # Physical values: scale, then mask the ODIM sentinels (NaN/Inf-safe).
    values = offset + gain * raw
    nodata_mask = np.isnan(raw) if np.isnan(nodata) else (raw == nodata)
    values[nodata_mask] = np.nan
    if np.isinf(undetect):
        values[np.isinf(raw)] = 0.0
    elif not np.isnan(undetect):
        values[raw == undetect] = 0.0

    # Swiss LV95 cell-centre coordinates. The grid is axis-aligned in the ODIM
    # projection, so transforming the upper-left corner gives the origin; row 0
    # is the northernmost row (y decreases with row index).
    transformer = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_proj4(projdef), always_xy=True)
    x0, y0 = transformer.transform(ul_lon, ul_lat)
    x = x0 + (np.arange(xsize) + 0.5) * xscale
    y = y0 - (np.arange(ysize) + 0.5) * yscale

    var_attrs = {"quantity": quantity}
    if long_name:
        var_attrs["long_name"] = long_name
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}, name=quantity.lower(), attrs=var_attrs)
    ds = da.to_dataset()
    ds.coords["x"].attrs.update({"units": "m", "long_name": "Swiss LV95 easting (CHX)"})
    ds.coords["y"].attrs.update({"units": "m", "long_name": "Swiss LV95 northing (CHY)"})
    ds.attrs.update({"projdef": projdef, "grid": "swiss_lv95", "odim_object": obj})
    if date and time:
        with contextlib.suppress(ValueError, IndexError):
            ts = np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}")
            ds = ds.assign_coords(time=ts)
    return ds


__all__ = ["open_composite"]
