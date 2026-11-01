# Contributing to foehn

Thanks for taking the time to contribute. Bug reports, documentation fixes,
new MeteoSwiss collections and code are all welcome.

## Finding something to work on

- **Bugs and feature requests** are tracked in
  [GitHub Issues](https://github.com/kayhendriksen/foehn/issues). Use the issue
  forms: a bug report asks for the output of `foehn version`, which answers most
  of the questions a maintainer would ask first.
- **Security issues** go through private vulnerability reporting, never a public
  issue. See [SECURITY.md](SECURITY.md).
- For anything larger than a small fix, open an issue first so the approach can
  be agreed before you spend time on it.

## Development setup

foehn is developed with [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
git clone https://github.com/kayhendriksen/foehn
cd foehn
uv sync          # creates .venv with foehn, every extra's dependencies, and the dev group
```

`uv sync` installs the `dev` dependency group by default — the test suite, the
type checker and the optional dependencies the gridded and MCP paths need. Run
anything inside that environment with `uv run`:

```bash
uv run foehn list
uv run python -c "import foehn; foehn.show_versions()"
```

Install the git hooks once. They run the same pinned ruff, ty, actionlint and
zizmor that CI runs, so a commit that passes locally passes the lint jobs too:

```bash
uv run pre-commit install
```

## Testing

```bash
uv run pytest                          # the full offline suite, in about ten seconds
uv run pytest tests/test_cli.py -q     # one module
uv run pytest --cov=foehn              # with the coverage floor CI enforces
uv run pytest -m live                  # opt-in tests against the live MeteoSwiss API
```

The default run never touches the network: the STAC API and CDNs are replaced by
an in-memory fetcher (`tests/fakes.py`). Tests marked `live` are deselected by
default and run weekly in CI against the real API.

Coverage is held at 97% branch coverage (`[tool.coverage.report]` in
`pyproject.toml`). New code comes with tests; a bug fix comes with a test that
fails without it.

## Linting, formatting and type checking

```bash
uv run pre-commit run --all-files   # everything CI's lint and typecheck jobs run
uv run ruff check --fix          # lint
uv run ruff format               # format
uv run ty check src/             # type check
```

foehn ships `py.typed`, so its annotations are a published interface: every
function in `src/` is annotated, and `ty` must pass.

## Documentation

The documentation lives in [`docs/`](docs/) and is published with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Preview it with:

```bash
uv run --group docs mkdocs serve
```

The API reference is generated from the docstrings, which follow the
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(enforced by ruff). [`docs/cli.md`](docs/cli.md) is written by hand, and
`tests/test_cli_docs.py` fails if a subcommand or flag is missing from it — add
the documentation in the same change as the flag.

## Architecture

Read [`CONTEXT.md`](CONTEXT.md) before changing anything structural. It is the
glossary of the words foehn uses — **Dataset**, **Collection**, **Dataset kind**,
**Workspace**, **Fetcher**, **Transfer**, **Reader** — and the record of why the
modules are split the way they are. In short:

| Layer | Modules |
|---|---|
| Front ends | `cli`, `mcp_server`, and the public `api` they share |
| Routing | `registry` picks each Dataset kind's download, convert, read and Zarr path |
| Pipelines | `downloads`, `convert`, `readers`, `grids`, `gridfiles`, `metadata` |
| Upstream conventions | `meteocsv`, `icon`, `odim`, `assets` |
| Infrastructure | `fetch` (every HTTP call), `transfer`, `state`, `workspace`, `atomicwrite`, `_locking` |
| Facts | `catalog`, `collections` |

`tests/test_layering.py` asserts these boundaries from the import graph — which
module may import which — so a change that crosses one fails a test rather than
review.

The CLI keeps two conventions from uv and ruff: a command's output goes to
stdout and every diagnostic goes to stderr, and the exit status is 0 for
success, 1 for a failure, 2 for a usage error.

## Pull requests

- Branch from `main`, and keep a pull request to one change.
- Name commits and pull requests with a conventional prefix: `fix:`, `feat:`,
  `docs:`, `ci:`, `chore:`, `build:` — for example
  `fix: forecast run selection skips incomplete runs`.
- Add a line to the **Unreleased** section of [`CHANGELOG.md`](CHANGELOG.md) for
  anything a user would notice, under *Breaking changes*, *New features* or
  *Bug fixes*.
- CI must be green: lint, typecheck, the test matrix (Python 3.11–3.15), the
  packaging smoke test, the docs build and the security scans.

## Releasing

Maintainers only.

1. Bump the version in `pyproject.toml`, `src/foehn/__init__.py` and
   `server.json` (`tests/test_version.py` checks they agree).
2. Move the **Unreleased** notes in `CHANGELOG.md` under a new
   `## X.Y.Z (YYYY-MM-DD)` heading.
3. Merge to `main`, then publish a GitHub release tagged `vX.Y.Z` with the same
   notes. `publish.yml` builds the wheel and sdist, signs them, publishes to
   PyPI through trusted publishing, attaches the SBOM and provenance to the
   release and updates the MCP Registry entry; `docs.yml` publishes the
   documentation site for the same release.

## Code of conduct

Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
