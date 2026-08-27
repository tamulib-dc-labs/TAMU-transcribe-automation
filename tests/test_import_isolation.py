"""No model stack may be imported at module level.

One rule:

    no module in src/asr may import torch or nemo at module level -
    only inside the function that needs it.

It keeps the base install to numpy, so the orchestrator, the queue tooling and
every test can run on a login node with no GPU stack present. Break it and
nothing fails locally, where NeMo is installed; it fails on a login node, or in
CI, at the moment something merely imports the package.

A static check rather than a runtime one, so the regression is caught where it
is written.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "asr"

#: Importing any of these drags in a model stack and its version pins.
HEAVY = {"torch", "nemo", "transformers", "torchaudio"}

#: The only third-party import allowed at module level anywhere in the package.
ALLOWED_THIRD_PARTY = {"numpy"}


def package_modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def top_level_imports(path: Path) -> set[str]:
    """Root package names imported at module scope (not inside a def/class)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:  # module scope only - nested imports are the point
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


# ------------------------------------------------------------------- static


@pytest.mark.parametrize("path", package_modules(), ids=lambda p: p.name)
def test_no_module_level_model_imports(path):
    """A heavy import here would break the venv that lacks that model."""
    offending = top_level_imports(path) & HEAVY
    assert not offending, (
        f"{path.relative_to(PACKAGE.parents[1])} imports {sorted(offending)} at module "
        "level. Move it inside the function that needs it, or importing this "
        "package will require a GPU stack that the login node does not have."
    )


def test_numpy_is_the_only_third_party_module_level_import():
    """Keeps the base install genuinely light; anything else needs a decision."""
    stdlib = set(sys.stdlib_module_names)
    extras: dict[str, set[str]] = {}
    for path in package_modules():
        names = top_level_imports(path)
        third_party = {
            n for n in names
            if n not in stdlib
            and not n.startswith("_")
            and n not in ALLOWED_THIRD_PARTY
            and n != "src"
        }
        if third_party:
            extras[str(path.relative_to(PACKAGE.parents[1]))] = third_party
    assert not extras, f"unexpected module-level third-party imports: {extras}"


# ------------------------------------------------------------------ runtime


def run_isolated(code: str) -> str:
    """Run code in a fresh interpreter so sys.modules is not pre-polluted."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PACKAGE.parents[1]),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_words_path_loads_no_model_stack():
    """What the Parakeet venv imports must not require transformers."""
    loaded = run_isolated(
        "import sys;"
        "from src.asr.parakeet import ParakeetBackend, ParakeetConfig;"
        "from src.asr.preprocess import prepare_audio;"
        "ParakeetBackend(ParakeetConfig());"
        "print(sorted(m for m in sys.modules if m in "
        "{'torch','nemo','transformers','torchaudio'}))"
    )
    assert loaded == "[]"


def test_the_diarization_path_loads_no_model_stack():
    loaded = run_isolated(
        "import sys;"
        "from src.asr.sortformer import SortformerDiarizer, SortformerConfig;"
        "SortformerDiarizer(SortformerConfig());"
        "print(sorted(m for m in sys.modules if m in "
        "{'torch','nemo','transformers','torchaudio'}))"
    )
    assert loaded == "[]"


def test_the_orchestrator_path_needs_no_model():
    """Filling the queue and fusing need no GPU stack at all."""
    loaded = run_isolated(
        "import sys;"
        "from src.asr.workqueue import FileWorkQueue;"
        "from src.asr.fusion import fuse;"
        "from src.asr.lines import build_lines;"
        "from src.asr.output import write_outputs;"
        "print(sorted(m for m in sys.modules if m in "
        "{'torch','nemo','transformers','torchaudio'}))"
    )
    assert loaded == "[]"


def test_constructing_a_backend_does_not_load_its_model():
    """Construction is cheap; only .load() pays the import cost."""
    out = run_isolated(
        "from src.asr.parakeet import ParakeetBackend, ParakeetConfig;"
        "from src.asr.sortformer import SortformerDiarizer, SortformerConfig;"
        "print(ParakeetBackend(ParakeetConfig())._model is None,"
        " SortformerDiarizer(SortformerConfig())._model is None)"
    )
    assert out == "True True"
