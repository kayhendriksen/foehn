"""Tests for upstream's ODIM-H5 radar conventions (foehn.odim).

A leaf module: one file in, one Dataset out, no fetcher and no workspace. These
go straight at it rather than through ``open_dataset``, which is what the split
bought — the scaling rules are the subject here, not the read path.
"""

import pytest
from conftest import write_odim_composite

pytest.importorskip("h5py")
pytest.importorskip("pyproj")
xr = pytest.importorskip("xarray")

from foehn.odim import open_composite  # noqa: E402


def test_a_polar_volume_is_refused(tmp_path):
    """The OGD products are Cartesian COMP images; a PVOL would be read as garbage."""
    path = tmp_path / "pvol.h5"
    write_odim_composite(path, obj="PVOL")

    with pytest.raises(ValueError, match="expected an ODIM 'COMP' composite"):
        open_composite(xr, path)


def test_a_finite_undetect_sentinel_becomes_zero(tmp_path):
    """'Nothing detected' is 0 mm of rain, not a missing reading — and it need not be Inf."""
    import numpy as np

    path = tmp_path / "cpc.h5"
    write_odim_composite(path, undetect=-1.0, nodata=float("nan"))

    ds = open_composite(xr, path)
    values = ds["acrr"].values

    assert values[1, 0] == 0.0
    assert not np.isnan(values[1, 0])


def test_the_meteoswiss_long_name_is_carried_onto_the_variable(tmp_path):
    """MeteoSwiss puts the human-readable name in /how/MeteoSwiss, outside the ODIM spec."""
    path = tmp_path / "cpc.h5"
    write_odim_composite(path, long_name="CombiPrecip hourly accumulation")

    ds = open_composite(xr, path)

    assert ds["acrr"].attrs["long_name"] == "CombiPrecip hourly accumulation"
    assert ds["acrr"].attrs["quantity"] == "ACRR"


def test_a_composite_with_no_long_name_carries_only_its_quantity(tmp_path):
    path = tmp_path / "cpc.h5"
    write_odim_composite(path)

    assert "long_name" not in open_composite(xr, path)["acrr"].attrs


def test_an_unparseable_stamp_leaves_the_composite_without_a_time(tmp_path):
    """A bad /what date must not fail an otherwise-good read — the grid is still usable."""
    path = tmp_path / "cpc.h5"
    write_odim_composite(path, date="not-a-date", time="......")

    assert "time" not in open_composite(xr, path).coords


def test_a_composite_with_no_stamp_at_all_carries_no_time(tmp_path):
    path = tmp_path / "cpc.h5"
    write_odim_composite(path, date="", time="")

    assert "time" not in open_composite(xr, path).coords


def test_a_nan_undetect_sentinel_zeroes_nothing(tmp_path):
    """Not every product declares 'nothing detected'; a NaN sentinel means there is no rule."""
    import numpy as np

    path = tmp_path / "cpc.h5"
    write_odim_composite(path, undetect=float("nan"), nodata=-999.0)

    values = open_composite(xr, path)["acrr"].values

    assert np.isnan(values[1, 0])  # the raw NaN is left as missing, not read as zero rain
    assert np.isnan(values[0, 2])  # -999 is the nodata sentinel here
    assert values[0, 1] == 1.5
