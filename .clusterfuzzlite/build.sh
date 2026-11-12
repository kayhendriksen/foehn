#!/bin/bash -eu
# Build one fuzzer per fuzz/fuzz_*.py for ClusterFuzzLite.
#
# foehn's runtime dependencies come from a hash-locked file, so the image installs
# exactly the reviewed bytes. foehn itself is not installed: PyInstaller (which
# compile_python_fuzzer drives) reads it straight from src/, the code under test.

python3 -m pip install --require-hashes --no-deps -r "$SRC/foehn/.clusterfuzzlite/requirements.txt"

for fuzzer in "$SRC"/foehn/fuzz/fuzz_*.py; do
  name="$(basename "$fuzzer" .py)"
  compile_python_fuzzer "$fuzzer" --paths="$SRC/foehn/src"
  # Seed corpus: real MeteoSwiss-shaped inputs give the fuzzer a head start.
  if [ -d "$SRC/foehn/fuzz/corpus/$name" ]; then
    (cd "$SRC/foehn/fuzz/corpus/$name" && zip -q -r "$OUT/${name}_seed_corpus.zip" .)
  fi
done
