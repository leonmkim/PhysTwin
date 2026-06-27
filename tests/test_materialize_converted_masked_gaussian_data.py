"""Unit tests for converted-session masked Gaussian materializer helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_process import materialize_converted_gaussian_data as base  # noqa: E402
from data_process import materialize_converted_masked_gaussian_data as M  # noqa: E402


def test_mask_stats_valid_synthetic():
    alpha = np.zeros((1080, 1920), dtype=np.float32)
    alpha[400:700, 800:1100] = 1.0
    stats = M.mask_stats(alpha)
    assert stats["valid"]
    assert stats["nonempty"]
    assert stats["not_full_frame"]


def test_mask_stats_rejects_empty():
    alpha = np.zeros((1080, 1920), dtype=np.float32)
    stats = M.mask_stats(alpha)
    assert not stats["valid"]


def test_mask_stats_rejects_full_frame():
    alpha = np.ones((1080, 1920), dtype=np.float32)
    stats = M.mask_stats(alpha)
    assert not stats["valid"]


def test_select_occlusion_prompt_results_picks_first_valid():
    results = [
        ("human", {"valid": False, "area_fraction": 0.0}),
        ("hand", {"valid": True, "area_fraction": 0.05}),
        ("arm", {"valid": True, "area_fraction": 0.08}),
    ]
    prompt, stats = M.select_occlusion_prompt_results(results)
    assert prompt == "hand"
    assert stats["area_fraction"] == 0.05


def test_select_occlusion_prompt_results_raises_when_none_valid():
    results = [
        ("human", {"valid": False, "area_fraction": 0.0}),
        ("hand", {"valid": False, "area_fraction": 0.9}),
    ]
    with pytest.raises(RuntimeError, match="No usable occlusion mask"):
        M.select_occlusion_prompt_results(results)


def test_overwrite_guard(tmp_path):
    case_dir = tmp_path / "case_a"
    case_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        base._prepare_case_dir(case_dir, overwrite=False)


def test_masked_materializer_rejects_traversal_before_file_io(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="case_name|case_dir"):
        M.materialize_converted_masked_gaussian_data(
            converted_session_path=tmp_path / "session",
            window_yaml=tmp_path / "w.yaml",
            camera_serials=["1", "2", "3"],
            anchor_serial="1",
            out_root=tmp_path / "out",
            case_name="../escape",
            shape_prior_glb=tmp_path / "missing.glb",
            cam0_object_mask=tmp_path / "missing_obj.png",
            cam0_occlusion_mask=tmp_path / "missing_occ.png",
            overwrite=True,
            write_overlays=False,
            projection_audit=False,
        )
