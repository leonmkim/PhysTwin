from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def test_parse_version_tuple():
    from data_process.o3d_utils import _parse_version_tuple

    assert _parse_version_tuple("1.26.4") == (1, 26, 4)
    assert _parse_version_tuple("4.11.0.86") == (4, 11, 0, 86)
    assert _parse_version_tuple("4.11.0") == (4, 11, 0)


def test_check_diagnostic_runtime_rejects_open3d_017():
    from data_process import o3d_utils

    mock_cv2 = mock.Mock()
    mock_cv2.__version__ = "4.11.0.86"
    with mock.patch.object(o3d_utils.o3d, "__version__", "0.17.0"):
        with mock.patch.object(o3d_utils.np, "__version__", "1.26.4"):
            with mock.patch.dict("sys.modules", {"cv2": mock_cv2}):
                with pytest.raises(RuntimeError, match="0.17"):
                    o3d_utils.check_diagnostic_runtime()


def test_create_checked_visualizer_raises_when_window_fails():
    from data_process.o3d_utils import create_checked_visualizer

    vis = mock.Mock()
    vis.create_window.return_value = False
    with mock.patch("data_process.o3d_utils.o3d.visualization.Visualizer", return_value=vis):
        with pytest.raises(RuntimeError, match="create_window failed"):
            create_checked_visualizer(width=64, height=48)


def test_create_mp4_writer_codec_fallback(tmp_path):
    from data_process.o3d_utils import create_mp4_writer, release_mp4_writer

    output = tmp_path / "probe.mp4"
    writer = mock.Mock()
    writer.isOpened.side_effect = [False, True]
    with mock.patch("cv2.VideoWriter", return_value=writer):
        with mock.patch(
            "cv2.VideoWriter_fourcc",
            side_effect=lambda *args: args,
        ):
            got, codec = create_mp4_writer(output, fps=1.0, width=16, height=16)
    assert codec == "mp4v"
    release_mp4_writer(got)


def test_visualize_pc_honors_disable_video_logging():
    import sys

    sys.modules.setdefault("pynput", mock.MagicMock())
    sys.modules.setdefault("pynput.keyboard", mock.MagicMock())
    from qqtt.utils import visualize as viz_mod
    from qqtt.utils.config import cfg

    cfg.disable_video_logging = True
    try:
        with mock.patch.object(viz_mod, "create_checked_visualizer") as create_vis:
            result = viz_mod.visualize_pc(
                np.zeros((2, 3, 3)),
                save_video=True,
                save_path="/tmp/should_not_render.mp4",
            )
        assert result is None
        create_vis.assert_not_called()
    finally:
        cfg.disable_video_logging = False


def test_render_existing_final_data_video_command_path(tmp_path):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))
    data_process_root = phystwin_root / "data_process"
    if str(data_process_root) not in sys.path:
        sys.path.insert(0, str(data_process_root))
    from data_process.data_process_sample import render_existing_final_data_video
    import pickle

    base = tmp_path / "data"
    case = "case_a"
    case_dir = base / case
    case_dir.mkdir(parents=True)
    track = {
        "object_points": np.zeros((2, 4, 3)),
        "object_colors": np.ones((2, 4, 3)),
        "object_visibilities": np.ones((2, 4), dtype=bool),
        "object_motions_valid": np.ones((2, 4), dtype=bool),
        "controller_points": np.zeros((2, 1, 3)),
    }
    pkl = case_dir / "final_data.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(track, f)
    out = tmp_path / "final_data.mp4"
    with mock.patch(
        "data_process.data_process_sample.visualize_track",
        return_value=None,
    ) as visualize_track:
        render_existing_final_data_video(
            base_path=str(base),
            case_name=case,
            output_path=out,
            final_data_pkl=pkl,
        )
    visualize_track.assert_called_once()
    assert visualize_track.call_args.kwargs["output_path"] == out


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Open3D probe requires Xvfb/DISPLAY",
)
def test_probe_open3d_calibrated_camera_under_xvfb_smoke():
    from data_process.o3d_utils import probe_open3d_calibrated_camera_under_xvfb

    result = probe_open3d_calibrated_camera_under_xvfb(width=32, height=24)
    assert result["pass"] is True
    assert result["codec"] in {"avc1", "mp4v"}
    assert result["frame_shape"] == [24, 32, 3]


@pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Open3D probe requires Xvfb/DISPLAY",
)
def test_probe_open3d_visualizer_under_xvfb_smoke():
    from data_process.o3d_utils import probe_open3d_visualizer_under_xvfb

    result = probe_open3d_visualizer_under_xvfb(width=32, height=24)
    assert result["pass"] is True
    assert result["codec"] in {"avc1", "mp4v"}


def test_render_inference_video_omits_gt_visibility_mask():
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    sys.modules.setdefault("pynput", mock.MagicMock())
    sys.modules.setdefault("pynput.keyboard", mock.MagicMock())

    import inference_warp as inf_mod

    vertices = np.zeros((2, 9864, 3))
    gt_visibility = np.ones((2, 6625), dtype=bool)

    with mock.patch.object(inf_mod, "load_visualization_cfg"):
        with mock.patch.object(
            inf_mod,
            "_load_final_data_arrays",
            return_value=(
                np.zeros((2, 6625, 3)),
                np.ones((2, 6625, 3)),
                np.zeros((2, 1, 3)),
                gt_visibility,
                np.ones((2, 6625), dtype=bool),
            ),
        ):
            with mock.patch.object(inf_mod, "visualize_pc") as visualize_pc:
                with mock.patch.object(inf_mod.pickle, "load", return_value=vertices):
                    with mock.patch("builtins.open", mock.mock_open(read_data=b"stub")):
                        inf_mod.render_inference_video_from_pkl(
                            base_path="/tmp/base",
                            case_name="case",
                            inference_pkl="/tmp/inference.pkl",
                            output_path="/tmp/out.mp4",
                        )

    assert visualize_pc.call_args[0][3] is None
    assert visualize_pc.call_args[0][4] is None
    assert vertices.shape[1] == 9864
    assert gt_visibility.shape[1] == 6625


def test_render_cma_rollout_video_only_init_uses_initial_parameters(tmp_path):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    import optimize_cma as cma_mod

    base = tmp_path / "data"
    case = "case_a"
    case_dir = base / case
    case_dir.mkdir(parents=True)
    (case_dir / "calibrate.pkl").write_bytes(pickle.dumps([np.eye(4)]))
    (case_dir / "metadata.json").write_text(
        json.dumps({"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "WH": [64, 48]})
    )
    (case_dir / "final_data.pkl").write_bytes(b"stub")
    out = tmp_path / "init.mp4"

    optimizer = mock.Mock()
    optimizer.build_initial_cma_parameters.return_value = np.zeros(12, dtype=np.float32)
    optimizer.render_rollout_video.return_value = {
        "output_path": str(out),
        "label": "init",
        "frame_count": 2,
        "status": "ok",
    }

    with mock.patch.object(cma_mod, "load_visualization_cfg"):
        with mock.patch.object(cma_mod, "OptimizerCMA", return_value=optimizer):
            result = cma_mod.render_cma_rollout_video_only(
                base_path=str(base),
                case_name=case,
                train_frame=2,
                rollout_kind="init",
                output_video_path=str(out),
                scratch_base_dir=str(tmp_path / "scratch"),
            )

    optimizer.build_initial_cma_parameters.assert_called_once()
    optimizer.optimal_results_to_cma_parameters.assert_not_called()
    optimizer.render_rollout_video.assert_called_once()
    assert result["label"] == "init"


def test_render_cma_rollout_video_only_optimal_loads_pickle(tmp_path):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    import optimize_cma as cma_mod

    base = tmp_path / "data"
    case = "case_a"
    case_dir = base / case
    case_dir.mkdir(parents=True)
    (case_dir / "calibrate.pkl").write_bytes(pickle.dumps([np.eye(4)]))
    (case_dir / "metadata.json").write_text(
        json.dumps({"intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "WH": [64, 48]})
    )
    (case_dir / "final_data.pkl").write_bytes(b"stub")
    optimal_path = tmp_path / "optimal_params.pkl"
    optimal = {"global_spring_Y": 1.0, "object_radius": 0.02}
    with open(optimal_path, "wb") as handle:
        pickle.dump(optimal, handle)
    out = tmp_path / "optimal.mp4"

    optimizer = mock.Mock()
    optimizer.optimal_results_to_cma_parameters.return_value = np.ones(12, dtype=np.float32)
    optimizer.render_rollout_video.return_value = {
        "output_path": str(out),
        "label": "optimal",
        "frame_count": 2,
        "status": "ok",
    }

    with mock.patch.object(cma_mod, "load_visualization_cfg"):
        with mock.patch.object(cma_mod, "OptimizerCMA", return_value=optimizer):
            result = cma_mod.render_cma_rollout_video_only(
                base_path=str(base),
                case_name=case,
                train_frame=2,
                rollout_kind="optimal",
                output_video_path=str(out),
                optimal_params_path=str(optimal_path),
                scratch_base_dir=str(tmp_path / "scratch"),
            )

    optimizer.optimal_results_to_cma_parameters.assert_called_once()
    optimizer.build_initial_cma_parameters.assert_not_called()
    assert result["label"] == "optimal"


def test_render_cma_rollout_video_only_missing_optimal_params(tmp_path):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    import optimize_cma as cma_mod

    with mock.patch.object(cma_mod, "load_visualization_cfg"):
        with pytest.raises(FileNotFoundError, match="optimal_params.pkl"):
            cma_mod.render_cma_rollout_video_only(
                base_path=str(tmp_path / "data"),
                case_name="case_a",
                train_frame=2,
                rollout_kind="optimal",
                output_video_path=str(tmp_path / "out.mp4"),
                optimal_params_path=str(tmp_path / "missing.pkl"),
                scratch_base_dir=str(tmp_path / "scratch"),
            )


def test_optimize_cma_cli_render_only_exits_before_optimize(tmp_path, monkeypatch):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    import optimize_cma as cma_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_cma.py",
            "--base_path",
            str(tmp_path / "data"),
            "--case_name",
            "case_a",
            "--train_frame",
            "2",
            "--render-cma-rollout-video-only",
            "--cma-rollout-kind",
            "init",
            "--output-video-path",
            str(tmp_path / "init.mp4"),
        ],
    )
    with mock.patch.object(cma_mod, "render_cma_rollout_video_only") as render_only:
        with mock.patch.object(cma_mod, "OptimizerCMA") as optimizer_cls:
            with pytest.raises(SystemExit) as exc:
                cma_mod.main()
    assert exc.value.code == 0
    render_only.assert_called_once()
    optimizer_cls.assert_not_called()


def test_optimize_cma_disable_video_logging_still_honored(monkeypatch):
    import sys

    phystwin_root = Path(__file__).resolve().parents[1]
    if str(phystwin_root) not in sys.path:
        sys.path.insert(0, str(phystwin_root))

    import optimize_cma as cma_mod
    from qqtt.utils.config import cfg

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_cma.py",
            "--base_path",
            "/tmp/data",
            "--case_name",
            "case_a",
            "--train_frame",
            "2",
            "--disable-video-logging",
        ],
    )
    optimizer = mock.Mock()
    with mock.patch.object(cma_mod, "load_visualization_cfg"):
        with mock.patch.object(cma_mod, "OptimizerCMA", return_value=optimizer):
            with mock.patch.object(cma_mod.logger, "set_log_file"):
                cma_mod.main()
    assert cfg.disable_video_logging is True
    optimizer.optimize.assert_called_once()


def test_gs_render_dynamics_view_indices_cli_parsing():
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "qqtt" / "utils" / "output_dirs.py"
    spec = importlib.util.spec_from_file_location("output_dirs_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.parse_view_indices_csv("0,50,100") == [0, 50, 100]
    assert module.parse_view_indices_csv("0,50,100,150") == [0, 50, 100, 150]
    assert module.parse_view_indices_csv(None) is None
    assert module.parse_view_indices_csv("") is None
    with pytest.raises(ValueError, match="non-negative"):
        module.parse_view_indices_csv("0,-1,50")
    with pytest.raises(ValueError, match="duplicate"):
        module.parse_view_indices_csv("0,50,50")
    with pytest.raises(ValueError, match="invalid view index 'abc'"):
        module.parse_view_indices_csv("0,abc,50")
