# The ClusterFuzzLite build image: OSS-Fuzz's Python builder (Python 3.11 with
# Atheris and PyInstaller), pinned by digest so the fuzzers build from known
# bytes. Dependabot's docker ecosystem bumps the digest.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:21f6fab17ace4dc76e1c56de91ee95d3d03da021df94b41056ba036f6c6f55e9
COPY . $SRC/foehn
WORKDIR $SRC/foehn
COPY .clusterfuzzlite/build.sh $SRC/
