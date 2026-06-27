"""Unit tests for converted-session Gaussian materializer case-name containment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_process import materialize_converted_gaussian_data as M  # noqa: E402


@pytest.mark.parametrize(
    "case_name",
    [
        "",
        ".",
        "..",
        "../escape",
        "foo/bar",
        "foo\\bar",
        "/tmp/escape",
        "../../escape",
    ],
)
def test_resolve_case_dir_rejects_unsafe_names(tmp_path, case_name):
    out_root = tmp_path / "out"
    out_root.mkdir()
    with pytest.raises(ValueError, match="case_name|case_dir"):
        M._resolve_case_dir(out_root, case_name)


def test_resolve_case_dir_accepts_simple_name(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    case_dir = M._resolve_case_dir(out_root, "f10c_masked_gaussian_materializer")
    assert case_dir == (out_root / "f10c_masked_gaussian_materializer").resolve()


def test_escape_case_name_does_not_delete_outside_sentinel(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    sentinel = tmp_path / "outside_sentinel"
    sentinel.mkdir()
    marker = sentinel / "KEEP"
    marker.write_text("keep")

    with pytest.raises(ValueError, match="case_name|case_dir"):
        M._resolve_case_dir(out_root, "../outside_sentinel")

    assert marker.exists()
    assert marker.read_text() == "keep"


def test_prepare_case_dir_only_runs_on_resolved_child(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    case_dir = M._resolve_case_dir(out_root, "safe_case")
    M._prepare_case_dir(case_dir, overwrite=False)
    assert case_dir.is_dir()
    assert case_dir.parent == out_root.resolve()
