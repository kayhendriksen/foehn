"""The Databricks bundle deploys the foehn it was deployed from, and says what it schedules.

``databricks bundle deploy`` used to upload whatever ``dist/foehn-*.whl`` matched:
nothing in the bundle built it and the deploy instructions never did either, so
a checkout deployed the last wheel anyone had built there — or two of them,
once the glob matched more than one version. And the deploy page said it
created two jobs while the bundle defined four, two of them every five minutes.
"""

import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "databricks.yml"
DOCS = ROOT / "docs" / "databricks.md"


@pytest.fixture(scope="module")
def bundle():
    yaml = pytest.importorskip("ruamel.yaml", reason="strict YAML parser not installed")
    return yaml.YAML(typ="safe").load(BUNDLE.read_text(encoding="utf-8"))


def _wheel_references(bundle) -> list[str]:
    """Every local wheel a job installs, from task libraries and serverless environments alike."""
    found: list[str] = []
    for job in bundle["resources"]["jobs"].values():
        for task in job.get("tasks", []):
            found += [lib["whl"] for lib in task.get("libraries", []) if "whl" in lib]
        for environment in job.get("environments", []):
            found += [dep for dep in environment["spec"].get("dependencies", []) if dep.endswith(".whl")]
    return found


def _out_dir(build: str) -> str:
    args = shlex.split(build)
    for flag in ("--out-dir", "-o"):
        if flag in args:
            return args[args.index(flag) + 1].rstrip("/")
    return "dist"


def test_the_bundle_builds_the_wheel_its_jobs_install(bundle):
    wheels = [artifact for artifact in bundle.get("artifacts", {}).values() if artifact.get("type") == "whl"]
    assert len(wheels) == 1, "the bundle must build foehn's wheel itself rather than ship whatever dist/ holds"
    build = wheels[0]["build"]
    args = shlex.split(build)
    assert args[:2] == ["uv", "build"]
    assert "--wheel" in args
    # A cleared output directory can only hold this build: a stale or second
    # version left there would match the jobs' glob alongside it.
    assert "--clear" in args

    out_dir = _out_dir(build)
    references = _wheel_references(bundle)
    assert references, "no job installs foehn's wheel"
    assert all(ref.startswith(f"./{out_dir}/") for ref in references), references


def test_the_wheel_is_built_somewhere_of_its_own(bundle):
    """``--clear`` empties its output directory, so it must not be the dist/ people build releases into."""
    (artifact,) = (a for a in bundle["artifacts"].values() if a.get("type") == "whl")
    assert _out_dir(artifact["build"]) != "dist"


def test_the_deploy_page_names_every_job_the_bundle_creates(bundle):
    docs = DOCS.read_text(encoding="utf-8")
    missing = [key for key in bundle["resources"]["jobs"] if f"`{key}`" not in docs]
    assert missing == []


def test_the_deploy_page_says_how_often_each_unpaused_job_runs(bundle):
    """A deploy to prod starts every unpaused schedule at once; the page has to say what that costs."""
    docs = DOCS.read_text(encoding="utf-8")
    for key, job in bundle["resources"]["jobs"].items():
        schedule = job.get("schedule", {})
        if schedule.get("pause_status") == "UNPAUSED":
            assert schedule["quartz_cron_expression"] in docs, f"{key}'s schedule is not documented"
