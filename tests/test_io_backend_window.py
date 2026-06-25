"""Tests for converted-session sync-index window consumption in ``io_backend``.

The data-free unit tests (YAML parsing, descriptor validation, ``create_io_backend``
precedence) always run. The integration test exercises the real converted-session
backend against an annotated ``phystwin_window.yaml`` and is skipped unless both the
session and ``dataset_converter.session_loader`` are importable; it asserts the backend's
exposed window matches the YAML's ``resolved_window`` / ``per_camera_resolution`` (so it
generalizes to any annotated session, not just the canonical one).
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_process import io_backend as M  # noqa: E402


def _write_window_yaml(path: Path, start, end, session_id: str = "session_x") -> Path:
    path.write_text(
        textwrap.dedent(
            f"""\
            schema_version: 1
            kind: phystwin_interaction_window
            session:
              session_id: {session_id}
            resolved_window:
              start_sync_index: {start}
              end_sync_index_exclusive: {end}
            """
        )
    )
    return path


def test_read_window_yaml_ok(tmp_path):
    p = _write_window_yaml(tmp_path / "w.yaml", 285, 374)
    assert M._read_window_yaml(p) == (285, 374)


def test_read_window_yaml_validation(tmp_path):
    bad_schema = tmp_path / "bad_schema.yaml"
    bad_schema.write_text("schema_version: 2\nkind: phystwin_interaction_window\n")
    with pytest.raises(ValueError):
        M._read_window_yaml(bad_schema)

    bad_kind = tmp_path / "bad_kind.yaml"
    bad_kind.write_text("schema_version: 1\nkind: something_else\n")
    with pytest.raises(ValueError):
        M._read_window_yaml(bad_kind)

    missing = tmp_path / "missing.yaml"
    missing.write_text("schema_version: 1\nkind: phystwin_interaction_window\n")
    with pytest.raises(ValueError):
        M._read_window_yaml(missing)

    inverted = _write_window_yaml(tmp_path / "inv.yaml", 374, 285)
    with pytest.raises(ValueError):
        M._read_window_yaml(inverted)


def test_descriptor_window_validation():
    desc = M.ConvertedSessionDescriptor
    # defaults are valid (no window)
    d = desc(session_path=Path("/x"), camera_serials=("12540311",))
    assert d.start_sync_index == 0 and d.end_sync_index_exclusive is None

    with pytest.raises(ValueError):
        desc(session_path=Path("/x"), camera_serials=("12540311",), start_sync_index=-1)
    with pytest.raises(ValueError):
        desc(
            session_path=Path("/x"),
            camera_serials=("12540311",),
            start_sync_index=10,
            end_sync_index_exclusive=10,
        )
    # window indices are native-grid; cannot combine with target_fps resampling
    with pytest.raises(ValueError):
        desc(
            session_path=Path("/x"),
            camera_serials=("12540311",),
            target_fps=15.0,
            start_sync_index=285,
            end_sync_index_exclusive=374,
        )


def test_create_io_backend_window_precedence(tmp_path, monkeypatch):
    captured: dict = {}

    class _StubBackend:
        def __init__(self, descriptor):
            captured["descriptor"] = descriptor

    monkeypatch.setattr(M, "ConvertedSessionBackend", _StubBackend)
    y = _write_window_yaml(tmp_path / "w.yaml", 285, 374)
    common = dict(
        io_backend="converted_session",
        converted_session_path="/x",
        camera_serials="12540311",
    )

    # YAML only
    M.create_io_backend(**common, window_yaml=str(y))
    d = captured["descriptor"]
    assert (d.start_sync_index, d.end_sync_index_exclusive) == (285, 374)

    # explicit indices override YAML
    M.create_io_backend(**common, window_yaml=str(y), start_sync_index=300, end_sync_index_exclusive=310)
    d = captured["descriptor"]
    assert (d.start_sync_index, d.end_sync_index_exclusive) == (300, 310)

    # neither -> default full range
    M.create_io_backend(**common)
    d = captured["descriptor"]
    assert d.start_sync_index == 0 and d.end_sync_index_exclusive is None


_SESSION = os.environ.get(
    "PHYSTWIN_TEST_SESSION",
    "/home/magna/belt_perception/converted_datasets/sessions/session_20260624_095554_523212",
)


def _session_and_loader_available() -> bool:
    if not Path(_SESSION).is_dir():
        return False
    try:
        import dataset_converter.session_loader  # noqa: F401
    except Exception:
        return False
    return (Path(_SESSION) / "metadata" / "phystwin_window.yaml").is_file()


@pytest.mark.skipif(
    not _session_and_loader_available(),
    reason="annotated converted session / dataset_converter.session_loader unavailable",
)
def test_window_matches_annotated_yaml():
    import yaml

    yaml_path = Path(_SESSION) / "metadata" / "phystwin_window.yaml"
    doc = yaml.safe_load(yaml_path.read_text())
    rw = doc["resolved_window"]
    serials = doc["timebase"]["anchor_serial"]
    all_serials = list(doc["per_camera_resolution"].keys())
    serial_csv = ",".join(all_serials)

    # no window -> full native overlap grid
    full = M.create_io_backend(
        io_backend="converted_session",
        converted_session_path=_SESSION,
        camera_serials=serial_csv,
        anchor_serial=serials,
    )
    assert full.frame_count() == full.raw_sync_sample_count()

    # windowed backend matches the YAML's resolved_window exactly
    b = M.create_io_backend(
        io_backend="converted_session",
        converted_session_path=_SESSION,
        camera_serials=serial_csv,
        anchor_serial=serials,
        window_yaml=str(yaml_path),
    )
    assert b.frame_count() == rw["frame_count"]
    assert b.window_start_sync_index() == rw["start_sync_index"]
    assert b.timestamp_ns(0) == rw["start_anchor_timestamp_ns"]
    assert b.timestamp_ns(rw["frame_count"] - 1) == rw["end_anchor_timestamp_ns"]

    # exposed frame 0 maps to the annotated start source frame of each camera
    for cam_id, serial in enumerate(all_serials):
        assert b.source_frame_index(cam_id, 0) == doc["per_camera_resolution"][serial]["start_source_frame_index"]

    # max_frames caps inside the window
    capped = M.create_io_backend(
        io_backend="converted_session",
        converted_session_path=_SESSION,
        camera_serials=serial_csv,
        anchor_serial=serials,
        window_yaml=str(yaml_path),
        max_frames=10,
    )
    assert capped.frame_count() == min(10, rw["frame_count"])
    assert capped.window_start_sync_index() == rw["start_sync_index"]

    # window indices cannot combine with target_fps resampling
    with pytest.raises(ValueError):
        M.create_io_backend(
            io_backend="converted_session",
            converted_session_path=_SESSION,
            camera_serials=serial_csv,
            anchor_serial=serials,
            window_yaml=str(yaml_path),
            target_fps=15.0,
        )
