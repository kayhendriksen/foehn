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
    """Sibling modules *module* imports from ``foehn``."""
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    found = {
        node.module.split(".")[1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("foehn.")
    }
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


@pytest.mark.parametrize("module", ["transfer", "grids", "readers", "api", "client"])
def test_nothing_but_the_registry_depends_on_parquet(module):
    assert "convert" not in _imports(module)


def test_meteocsv_is_the_bottom_of_the_read_stack():
    """Upstream's file conventions know about datasets and nothing else."""
    assert _imports("meteocsv") <= {"collections"}


@pytest.mark.parametrize("module", ["collections", "workspace", "_urls"])
def test_the_leaf_modules_import_no_foehn(module):
    """Dataset facts, the layout and URL validation sit under everything."""
    assert _imports(module) == set()
