"""Locate the separately installed Perception Encoder source."""

import importlib
import os
from pathlib import Path
import sys


def load_perception_encoder():
    repo = Path(__file__).resolve().parents[2]
    explicit = os.environ.get("PE_ROOT")
    candidates = ([Path(explicit).expanduser()] if explicit else
                  [repo.parent / "perception_models", repo / "perception_models"])
    for root in candidates:
        if (root / "core/vision_encoder/pe.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            importlib.invalidate_caches()
            break
    else:
        if explicit:
            raise ImportError(
                f"Invalid PE_ROOT={explicit!r}: expected core/vision_encoder/pe.py under it."
            )
    try:
        return importlib.import_module("core.vision_encoder.pe")
    except ModuleNotFoundError as exc:
        if exc.name not in ("core", "core.vision_encoder", "core.vision_encoder.pe"):
            raise
        raise ImportError(
            "Perception Encoder source is not importable. Set PE_ROOT to the "
            "perception_models directory containing core/vision_encoder/pe.py, "
            "or clone perception_models beside this repository. "
            "Do not install an unrelated package named 'core'."
        ) from exc
