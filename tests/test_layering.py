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


@pytest.mark.parametrize("module", ["collections", "workspace", "_urls", "archives", "odim", "atomicwrite"])
def test_the_leaf_modules_import_no_foehn(module):
    """Dataset facts, the layout, URL validation, the ZIP guards, ODIM and staging sit under everything.

    ``odim`` is upstream's radar file format and nothing else: one path in, one
    Dataset out. It takes ``xr`` as an argument rather than importing it, so it
    does not even reach the optional-dependency guard.
    """
    assert _imports(module) == set()


def test_the_grid_readers_do_not_know_how_files_arrive():
    """``grids`` opens what it is handed; listing and fetching are ``gridfiles``.

    All three used to be one module, so the reader that parses an ODIM composite
    imported the download engine.
    """
    deps = _imports("grids")
    assert "transfer" not in deps
    assert "assets" not in deps
    assert "gridfiles" not in deps


def test_the_run_state_does_not_depend_on_the_download_engine():
    """``state`` imported ``transfer`` for one filesystem primitive, which is ``atomicwrite`` now.

    The same shape as ``transfer`` importing ``convert``: a module reaching into
    a pipeline stage for a helper that was never that stage's to own.
    """
    assert _imports("state") == {"atomicwrite", "workspace"}


def test_every_write_to_the_workspace_stages():
    """Four modules used to state the temp-file-then-replace rule; now they call it.

    Guards the rule itself rather than the imports: a fifth writer that hand-rolls
    ``.replace`` is exactly the regression this split was for.
    """
    hand_rolled = {
        module
        for module in ("transfer", "convert", "fetch", "state", "downloads", "gridfiles")
        if ".replace(path)" in (SRC / f"{module}.py").read_text(encoding="utf-8")
    }
    assert hand_rolled == set()


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
    """Reached through ``api``'s three wrappers, like every other stage through its seam."""
    importers = {module for module, deps in _graph().items() if "metadata" in deps}
    assert importers == {"api"}


def test_the_grid_fetching_has_one_consumer():
    """Like ``convert`` and ``downloads``: reached through the registry, not called directly."""
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
    source = (SRC / "registry.py").read_text(encoding="utf-8")
    assert "sanitize" not in source
    assert "require_dask" not in source
    assert ".chunk(" not in source


def test_the_load_path_reads_no_csv_itself():
    """Every CSV the load path parses is parsed by ``meteocsv``.

    The indoor archive's members were the exception: their separator and schema
    window were spelled out in ``readers`` and again in ``convert``, above the
    module that owns upstream's conventions, while every other kind already had
    an eager reader and a lazy scanner down there.
    """
    source = (SRC / "readers.py").read_text(encoding="utf-8")
    assert "separator=" not in source
    assert "read_csv" not in source


def test_the_convert_stage_states_no_indoor_csv_conventions():
    """The other half of the same duplication. Its remaining ``separator=`` literals
    are the standard kind's, wrapped in the dtype-drift retry, and the normals TXT.
    """
    assert 'separator=","' not in (SRC / "convert.py").read_text(encoding="utf-8")
