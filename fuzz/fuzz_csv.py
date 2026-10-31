"""Fuzz the MeteoSwiss CSV decoders and parsers with arbitrary bytes.

Every CSV foehn reads comes from upstream, so these see whatever a compromised
or simply broken MeteoSwiss file contains. The decoders are total — they must
return for any input — and the parsers may only fail with the errors they
document: ValueError, or a Polars parse error.
"""

import contextlib
import sys

try:
    import atheris
except ImportError:  # tests/test_fuzz_targets.py runs these without Atheris
    atheris = None

with atheris.instrument_imports(include=["foehn"]) if atheris else contextlib.nullcontext():
    import polars as pl

    from foehn import meteocsv

EXPECTED = (ValueError, pl.exceptions.PolarsError)


def TestOneInput(data: bytes) -> None:  # noqa: N802 - the name libFuzzer and Atheris call
    text = meteocsv.decode_meteoswiss_csv(data)
    assert isinstance(text, str)
    # bytes, or a zero-copy memoryview when the input starts with a BOM.
    normalised = meteocsv.utf8_meteoswiss_csv(data)
    assert isinstance(normalised, (bytes, memoryview))
    assert isinstance(meteocsv.parse_metadata_types(data), dict)
    meteocsv.column_from_dtype_error(text)

    for parse in (
        lambda: meteocsv.parse_csv_bytes(data, {}),
        # What foehn.load() actually does with a downloaded body.
        lambda: meteocsv.parse_csv_bytes(normalised, {}),
        lambda: meteocsv.parse_forecast_points(data),
        lambda: meteocsv.parse_climate_scenarios_csv(data, "ogd-climate-scenarios-ch2025_abe_pr_gwl1.5.csv"),
    ):
        with contextlib.suppress(*EXPECTED):
            parse()


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
