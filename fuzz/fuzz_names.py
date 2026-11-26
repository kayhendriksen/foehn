"""Fuzz the parsers that read meaning out of upstream names, URLs and timestamps.

Asset hrefs, filenames and STAC timestamps all arrive from the listing. The
filename parsers are total; URL validation either hands the URL back or raises
ValueError, and anything it accepts must really be https on an allowed host.
"""

import contextlib
import sys
from urllib.parse import urlparse

try:
    import atheris
except ImportError:  # tests/test_fuzz_targets.py runs these without Atheris
    atheris = None

with atheris.instrument_imports(include=["foehn"]) if atheris else contextlib.nullcontext():
    from foehn import _urls, collections, state


def TestOneInput(data: bytes) -> None:  # noqa: N802 - the name libFuzzer and Atheris call
    text = data.decode("utf-8", errors="replace")

    for parse in (
        collections.time_slice_from_filename,
        collections.granularity_from_filename,
        collections.forecast_run_from_filename,
        collections.forecast_parameter_from_filename,
    ):
        result = parse(text)
        assert result is None or isinstance(result, str)
    assert isinstance(_urls.asset_filename(text), str)

    for validate, hosts in (
        (_urls.validate_stac_url, _urls.STAC_DOMAINS),
        (_urls.validate_download_href, _urls.DOWNLOAD_DOMAINS),
    ):
        try:
            accepted = validate(text)
        except ValueError:
            continue
        parsed = urlparse(accepted)
        assert parsed.scheme == "https", accepted
        assert parsed.hostname in hosts, accepted

    instant = state.parse_instant(text)
    assert instant is None or instant.tzinfo is not None


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
