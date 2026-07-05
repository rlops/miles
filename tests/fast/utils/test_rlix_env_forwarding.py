"""Source-level checks that operator env knobs survive the Ray runtime_env boundary.

The tms preload-on-Blackwell guard relaxation (cu13+ wheels allowed) added two
new knobs — MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL and MILES_MAX_RESIDUAL_GPU_MEM_GB —
that only work if the rlix example drivers forward them into each pipeline's
runtime_env, and the legacy actor_group path forwards MILES_TMS_HOOK_MODE like
the placement path does (codex review requirement on the guard-relaxation PR).

These tests read/AST-parse the source files instead of importing them: the
example drivers pull in ray/miles heavyweight deps that fast tests must avoid.
"""

import ast
from pathlib import Path

# tests/fast/utils/test_rlix_env_forwarding.py -> parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

RUN_MILES_DUAL = REPO_ROOT / "examples" / "rlix" / "run_miles_dual.py"
RUN_MILES_RLIX = REPO_ROOT / "examples" / "rlix" / "run_miles_rlix.py"
ACTOR_GROUP = REPO_ROOT / "miles" / "ray" / "actor_group.py"

REQUIRED_FORWARDED_KEYS = (
    "MILES_TMS_HOOK_MODE",
    "MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL",
    "MILES_MAX_RESIDUAL_GPU_MEM_GB",
)


def _forwarding_tuples(path: Path) -> list[set[str]]:
    """All-string-constant tuples in ``path`` that mention MILES_TMS_HOOK_MODE."""
    tree = ast.parse(path.read_text(), filename=str(path))
    tuples = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple):
            continue
        values = {el.value for el in node.elts if isinstance(el, ast.Constant) and isinstance(el.value, str)}
        if len(values) == len(node.elts) and "MILES_TMS_HOOK_MODE" in values:
            tuples.append(values)
    return tuples


def _function_sources(path: Path) -> dict[str, str]:
    """Map function/method name -> source segment for every def in ``path``."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    return {
        node.name: ast.get_source_segment(source, node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_run_miles_dual_forwards_tms_knobs():
    tuples = _forwarding_tuples(RUN_MILES_DUAL)
    assert tuples, f"no env-forwarding tuple found in {RUN_MILES_DUAL}"
    for keys in tuples:
        for required in REQUIRED_FORWARDED_KEYS:
            assert required in keys, f"{required} missing from forwarding tuple in {RUN_MILES_DUAL}"


def test_run_miles_rlix_forwards_tms_knobs():
    tuples = _forwarding_tuples(RUN_MILES_RLIX)
    assert tuples, f"no env-forwarding tuple found in {RUN_MILES_RLIX}"
    for keys in tuples:
        for required in REQUIRED_FORWARDED_KEYS:
            assert required in keys, f"{required} missing from forwarding tuple in {RUN_MILES_RLIX}"


def test_actor_group_forwards_tms_knobs_on_both_paths():
    # Both LD_PRELOAD injection blocks (placement + legacy) must forward
    # MILES_TMS_HOOK_MODE (or actors on the legacy path silently fall back to
    # preload and trip the Blackwell guard) AND the guard's escape hatch
    # MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL (read inside the actor process, so
    # it is inert under Ray runtime_env isolation unless forwarded).
    functions = _function_sources(ACTOR_GROUP)
    for func_name in ("_allocate_gpus_via_placements", "_allocate_gpus_for_actor"):
        assert func_name in functions, f"{func_name} not found in {ACTOR_GROUP}"
        for env_key in ("MILES_TMS_HOOK_MODE", "MILES_TMS_ALLOW_PRELOAD_ON_BLACKWELL"):
            assert (
                env_key in functions[func_name]
            ), f"{func_name} in {ACTOR_GROUP} does not forward {env_key}"
