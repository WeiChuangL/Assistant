import importlib.util
import sys
from pathlib import Path

from src.skill.registry import skill_registry


_SKILLS_DIR = Path(__file__).parent / "skills"


def load_skills():
    """Scan skills/ directory and register all .py files with a SKILL dict."""
    if not _SKILLS_DIR.exists():
        return

    for file_path in _SKILLS_DIR.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        _load_skill_file(file_path)


def _load_skill_file(path: Path):
    module_name = f"_skill_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return

    skill_def = getattr(module, "SKILL", None)
    if isinstance(skill_def, dict) and "name" in skill_def:
        skill_registry.register(skill_def)
