"""The fuzz targets in fuzz/ run here too, without Atheris, over seeds and mutations.

ClusterFuzzLite builds and runs them coverage-guided in CI. This keeps every
harness importable and correct in the ordinary test run, and exercises each
property on a few hundred inputs: a seed corpus of real MeteoSwiss-shaped data
plus deterministic random mutations of it.
"""

import importlib.util
import random
from pathlib import Path

import pytest

FUZZ = Path(__file__).resolve().parent.parent / "fuzz"
TARGETS = sorted(FUZZ.glob("fuzz_*.py"))
MUTATIONS = 300


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mutate(rng: random.Random, data: bytes) -> bytes:
    out = bytearray(data)
    for _ in range(rng.randint(1, 8)):
        choice = rng.random()
        if choice < 0.4 and out:
            out[rng.randrange(len(out))] = rng.randrange(256)
        elif choice < 0.7:
            at = rng.randrange(len(out) + 1)
            out[at:at] = rng.randbytes(rng.randint(1, 6))
        elif out:
            at = rng.randrange(len(out))
            del out[at : at + rng.randint(1, 20)]
    return bytes(out)


def test_every_target_has_a_seed_corpus():
    assert TARGETS
    for target in TARGETS:
        assert any((FUZZ / "corpus" / target.stem).iterdir()), f"{target.stem} has no seeds"


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: p.stem)
def test_target_holds_its_property_on_seeds_and_mutations(target):
    test_one_input = _load(target).TestOneInput
    seeds = [path.read_bytes() for path in sorted((FUZZ / "corpus" / target.stem).iterdir())]
    rng = random.Random(target.stem)  # noqa: S311 - deterministic mutations, not secrets

    for data in [b"", *seeds]:
        test_one_input(data)
    for _ in range(MUTATIONS):
        source = rng.choice(seeds) if rng.random() < 0.8 else rng.randbytes(rng.randint(0, 64))
        test_one_input(_mutate(rng, source))
