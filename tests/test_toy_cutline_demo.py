from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "examples" / "toy_cutline_demo.py"


def _load_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("toy_cutline_demo", DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the toy demo module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_writes_success_failure_and_sanitized_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo = _load_demo()
    monkeypatch.chdir(tmp_path)

    assert demo.main(["--output-dir", "demo_output"]) == 0
    stdout = capsys.readouterr().out
    assert "Successful toy case" in stdout
    assert "Structured failure toy case" in stdout
    assert "no_calyx" in stdout

    output = tmp_path / "demo_output"
    summary_text = (output / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert str(tmp_path.resolve()) not in summary_text
    assert summary["class_mapping"] == {"background": 0, "Flesh": 1, "Calyx": 2}
    assert summary["machine_calibrated"] is False
    assert summary["demonstrates_segmentation_model_accuracy"] is False
    assert summary["matched_occlusion_figure"] == {
        "occluder_area_pixels": 116,
        "provenance": ("procedural toy data and repository matched-placement code"),
        "regions": ["attachment", "calyx_tip", "flesh_far", "background"],
        "same_template_translated": True,
        "seed": 0,
        "severity_fraction": 0.02,
        "visualization": "matched_occlusion_regions.svg",
    }
    assert summary["cases"]["successful_coordinate"]["v2a"]["status"] == "ok"
    assert summary["cases"]["successful_coordinate"]["v2b"]["status"] == "ok"
    assert summary["cases"]["insufficient_evidence"]["v2a"] == {
        "coordinate_pixels": None,
        "failure_code": "no_calyx",
        "status": "failed",
    }
    assert summary["cases"]["insufficient_evidence"]["v2b"]["status"] == ("failed")

    for name in ("successful_coordinate.png", "insufficient_evidence.png"):
        with Image.open(output / name) as visualization:
            assert visualization.mode == "RGB"
            assert visualization.width > 0
            assert visualization.height > 0
    matched_svg = (output / "matched_occlusion_regions.svg").read_text(encoding="utf-8")
    assert "not a real fruit image" in matched_svg
    assert matched_svg.count('<use href="#occluder"') == 4
    assert "base64" not in matched_svg

    for make_case in (
        demo.make_success_case,
        demo.make_insufficient_evidence_case,
    ):
        image, mask = make_case()
        assert image.shape == (*mask.shape, 3)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1, 2}


def test_demo_refuses_overwrite_and_paths_outside_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _load_demo()
    monkeypatch.chdir(tmp_path)
    demo.run_demo("demo_output")

    with pytest.raises(FileExistsError, match="--overwrite"):
        demo.run_demo("demo_output")
    replaced = demo.run_demo("demo_output", overwrite=True)
    assert replaced["cases"]["successful_coordinate"]["v2a"]["status"] == "ok"

    with pytest.raises(ValueError, match="relative path"):
        demo.run_demo(tmp_path / "absolute_output")
    with pytest.raises(ValueError, match="inside the current working directory"):
        demo.run_demo(Path("..") / "escaped_output")


def test_demo_execution_path_does_not_import_torch(tmp_path: Path) -> None:
    probe = """
import importlib.abc
import runpy
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"toy demo imported {fullname}")
        return None

sys.meta_path.insert(0, BlockTorch())
sys.argv = [sys.argv[1], "--output-dir", "torch_free_output"]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as error:
    if error.code not in (0, None):
        raise
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(DEMO_PATH)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Structured failure toy case" in completed.stdout
