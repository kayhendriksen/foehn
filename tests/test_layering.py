"""What each module in ``foehn`` is allowed to depend on.

Asserted from the import graph rather than from a diagram, because the defect
this guards against is invisible in any single file: ``transfer`` — the download
engine — imported ``convert`` for one byte-level helper, so every module that
used ``transfer`` depended on the Parquet conversion stage, including the
gridded *read* path.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "foehn"


def _imports(module: str) -> set[str]:
    """Sibling modules *module* imports from ``foehn``.

    Both spellings count: ``from foehn.icon import attach_lonlat`` names the
    module in ``node.module``, ``from foehn import icon`` names it in the alias.
    Reading only the first would have let the second open any edge it liked.
    """
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.startswith("foehn."):
            found.add(node.module.split(".")[1])
        elif node.module == "foehn":
            found |= {alias.name for alias in node.names}
    return found - {module}


def _calls(module: str) -> set[str]:
    """Every function or method name called in *module*.

    Read from the tree rather than matched in the text: ``"dask" in source`` was
    true of a comment, and ``".chunk(" in source`` would be true of a docstring.
    A rule worth asserting is worth asserting about the code.
    """
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _keyword_values(module: str, keyword: str) -> set[object]:
    """Every constant passed as *keyword* at a call in *module*."""
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    found: set[object] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == keyword and isinstance(kw.value, ast.Constant):
                found.add(kw.value.value)
    return found


def _renames_a_path(module: str) -> bool:
    """Whether *module* calls ``<path>.replace(target)`` — the move half of staging.

    One positional argument and no keywords is what tells it apart from
    ``str.replace``, which takes at least two. The substring this replaces could
    not, which is why it was spelled ``.replace(path)`` and would have missed
    ``.replace(self.target)``.
    """
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and len(node.args) == 1
        and not node.keywords
        for node in ast.walk(tree)
    )


def _graph() -> dict[str, set[str]]:
    return {p.stem: _imports(p.stem) for p in SRC.glob("*.py") if p.stem != "__init__"}


def test_the_import_graph_is_acyclic():
    """A cycle is what forced the load path through an if-ladder in ``api`` once."""
    graph = _graph()
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node: str, trail: list[str]) -> None:
        if node in done:
            return
        if node in visiting:
            raise AssertionError(f"import cycle: {' -> '.join([*trail, node])}")
        visiting.add(node)
        for dep in sorted(graph.get(node, ())):
            walk(dep, [*trail, node])
        visiting.discard(node)
        done.add(node)

    for module in sorted(graph):
        walk(module, [])


def test_the_convert_stage_has_exactly_one_consumer():
    """``convert`` is one pipeline stage, reached through the registry like the others.

    Five modules used to import it; four wanted only the CSV conventions, which
    are ``meteocsv`` now.
    """
    importers = {module for module, deps in _graph().items() if "convert" in deps}
    assert importers == {"registry"}


@pytest.mark.parametrize("module", ["transfer", "grids", "readers", "api", "downloads"])
def test_nothing_but_the_registry_depends_on_parquet(module):
    assert "convert" not in _imports(module)


def test_the_load_path_does_not_import_a_download_module():
    """``readers`` needed one ZIP guard and imported the whole download module for it."""
    assert "downloads" not in _imports("readers")
    assert "archives" in _imports("readers")


def test_the_download_adapters_have_one_consumer():
    """Like ``convert``: reached through the registry row, not called directly."""
    importers = {module for module, deps in _graph().items() if "downloads" in deps}
    assert importers == {"registry"}


def test_meteocsv_is_the_bottom_of_the_read_stack():
    """Upstream's file conventions know about datasets and nothing else."""
    assert _imports("meteocsv") <= {"collections"}


@pytest.mark.parametrize("module", ["catalog", "workspace", "_urls", "archives", "odim", "_locking", "docstrings"])
def test_the_leaf_modules_import_no_foehn(module):
    """Dataset facts, the layout, URL validation, the ZIP guards, ODIM, staging and
    docstring rendering sit under everything.

    ``odim`` is upstream's radar file format and nothing else: one path in, one
    Dataset out. It takes ``xr`` as an argument rather than importing it, so it
    does not even reach the optional-dependency guard.
    """
    assert _imports(module) == set()


def test_the_grid_reader_owns_file_acquisition_without_absorbing_transfer():
    """The Grid reader sequences an injected acquisition Adapter without importing it."""
    deps = _imports("grids")
    assert "transfer" not in deps
    assert "assets" not in deps
    assert "gridfiles" not in deps


def test_the_run_state_does_not_depend_on_the_download_engine():
    """``state`` imported ``transfer`` for one filesystem primitive, which is ``atomicwrite`` now.

    The same shape as ``transfer`` importing ``convert``: a module reaching into
    a pipeline stage for a helper that was never that stage's to own.
    """
    assert _imports("state") == {"_locking", "atomicwrite", "workspace"}


def test_every_write_to_the_workspace_stages():
    """Four modules used to state the temp-file-then-replace rule; now they call it.

    Guards the rule itself rather than the imports: a fifth writer that hand-rolls
    ``.replace`` is exactly the regression this split was for.
    """
    writers = ("transfer", "convert", "fetch", "state", "downloads", "gridfiles")
    hand_rolled = {module for module in writers if _renames_a_path(module)}
    assert hand_rolled == set()
    # And the rule has somewhere to live: the leaf that owns it does the move.
    assert _renames_a_path("atomicwrite")


def test_the_public_surface_only_delegates():
    """``api`` states the public contract; every stage below it is reached through a seam.

    It used to hold one stage outright — the metadata tables, fetched, decoded and
    renamed inline — and to pick cube-vs-single itself in ``to_zarr``, which is
    why it reached into ``assets``, ``meteocsv`` and ``grids`` while ``mcp_server``,
    a peer front end, needed only the registry.
    """
    forbidden = {"assets", "meteocsv", "grids", "gridfiles", "downloads", "convert", "icon", "odim"}
    assert _imports("api") & forbidden == set()


def test_the_metadata_tables_have_one_consumer():
    """The public wrappers and MCP Adapter share the authoritative published schema."""
    importers = {module for module, deps in _graph().items() if "metadata" in deps}
    assert importers == {"api", "mcp_server"}


def test_the_grid_fetching_has_one_consumer():
    """Grid file acquisition sits behind the Grid reader Seam."""
    importers = {module for module, deps in _graph().items() if "gridfiles" in deps}
    assert importers == {"registry"}


def test_upstreams_grid_conventions_have_one_consumer_each():
    """ODIM's scaling and ICON's cell coordinates are read by the grid readers, and nothing else."""
    graph = _graph()
    for module in ("odim", "icon"):
        assert {m for m, deps in graph.items() if module in deps} == {"grids"}


def test_the_registry_routes_a_zarr_write_without_knowing_the_recipe():
    """Which writer is the registry's; what a Dataset needs on the way is ``grids``'.

    ``sanitize_noncf_time_units`` and ``require_dask`` were exported only so the
    routing table could sequence them around the write — and ``cube_grib2``
    stated the same steps a second time inside ``grids`` on its own way out.
    Guards the rule rather than the imports: restating either step above the
    seam is the regression, however it is spelled.
    """
    called = _calls("registry")
    assert not {name for name in called if "sanitize" in name}
    assert "require_dask" not in called
    assert "chunk" not in called


def test_the_load_path_reads_no_csv_itself():
    """Every CSV the load path parses is parsed by ``meteocsv``.

    The indoor archive's members were the exception: their separator and schema
    window were spelled out in ``readers`` and again in ``convert``, above the
    module that owns upstream's conventions, while every other kind already had
    an eager reader and a lazy scanner down there.
    """
    called = _calls("readers")
    assert not called & {"read_csv", "scan_csv"}
    assert _keyword_values("readers", "separator") == set()


def test_the_convert_stage_states_no_upstream_csv_conventions():
    """Standard, preamble, archive and normals parsing all belong to ``meteocsv``."""
    assert _keyword_values("convert", "separator") == set()


def test_delta_ingest_uses_the_shared_meteoswiss_csv_implementation():
    """The independent Delta entry point must publish the same normalized frames.

    This script used to parse each Dataset kind directly with Polars, so Parquet,
    Reader and Delta could disagree while every package-module layering test
    remained green. Inspecting the entry point keeps it on the same Interface.
    """
    path = SRC.parents[1] / "scripts" / "ingest_delta.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    polars_csv_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr in {"read_csv", "scan_csv"}
    }
    shared_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "foehn.meteocsv"
        for alias in node.names
    }

    assert polars_csv_calls == set()
    assert {"read_dataset_csv", "scan_dataset_csv", "read_metadata_csv", "read_normals_txt"} <= shared_imports
