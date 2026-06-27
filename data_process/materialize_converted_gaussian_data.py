"""Materialize native-resolution QQTT Gaussian data from a converted session.

Writes a case directory compatible with ``readQQTTSceneInfo`` and
``gaussian_splatting/generate_interp_poses.py`` without calling
``export_gaussian_data.py`` or fabricating masks.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_process.io_backend import (
    WORLD_TRANSFORM_CHOICES,
    WORLD_TRANSFORM_PHYS_TWIN_Z_UP,
    create_io_backend,
)

SCRIPT_VERSION = "1.0.0"
QQTT_CAMERA_COUNT = 3


def _best_effort_git_head(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize native-resolution QQTT gaussian_data from a converted session."
        ),
    )
    parser.add_argument(
        "--converted-session-path",
        type=Path,
        required=True,
        help="Converted session root (dataset_converter session directory).",
    )
    parser.add_argument(
        "--window-yaml",
        type=Path,
        required=True,
        help="PhysTwin interaction window YAML (phystwin_window.yaml).",
    )
    parser.add_argument(
        "--camera-serials",
        nargs="+",
        required=True,
        help="ZED serials mapped to camera ids 0..N-1 in listed order.",
    )
    parser.add_argument(
        "--anchor-serial",
        type=str,
        required=True,
        help="Anchor camera serial for native sync indexing.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; case is written to <out-root>/<case-name>/.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        required=True,
        help="Case subdirectory name under --out-root.",
    )
    parser.add_argument(
        "--shape-prior-glb",
        type=Path,
        required=True,
        help="Source mesh GLB copied to <case>/shape_prior.glb.",
    )
    parser.add_argument(
        "--world-transform",
        type=str,
        default=WORLD_TRANSFORM_PHYS_TWIN_Z_UP,
        choices=list(WORLD_TRANSFORM_CHOICES),
        help=f"Rigid c2w remap for converted session (default: {WORLD_TRANSFORM_PHYS_TWIN_Z_UP}).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Window-relative frame index for RGB/depth (default: 0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output case directory.",
    )
    return parser.parse_args()


def _validate_camera_serials(serials: list[str]) -> list[str]:
    cleaned = [str(s).strip() for s in serials if str(s).strip()]
    if len(cleaned) < 2:
        raise ValueError(
            f"At least 2 camera serials are required; got {len(cleaned)}"
        )
    if len(cleaned) != QQTT_CAMERA_COUNT:
        raise ValueError(
            f"QQTT Gaussian materialization requires exactly {QQTT_CAMERA_COUNT} "
            f"cameras; got {len(cleaned)} serial(s). "
            "generate_interp_poses.py and the current training path expect three views."
        )
    return cleaned


def _prepare_case_dir(case_dir: Path, overwrite: bool) -> None:
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output case already exists: {case_dir}. Pass --overwrite to replace."
            )
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)


def _write_rgb_png(rgb: np.ndarray, path: Path) -> tuple[int, int]:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got shape {rgb.shape}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write PNG: {path}")
    loaded = cv2.imread(str(path))
    if loaded is None:
        raise RuntimeError(f"Failed to read back PNG: {path}")
    if loaded.shape[:2] != (height, width):
        raise RuntimeError(
            f"PNG round-trip shape mismatch for {path}: "
            f"expected {(height, width)}, got {loaded.shape[:2]}"
        )
    return height, width


def _write_depth_mm(depth_mm: np.ndarray, path: Path) -> tuple[int, int]:
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {depth.shape}")
    np.save(path, depth)
    loaded = np.load(path)
    if loaded.shape != depth.shape or loaded.dtype != np.float32:
        raise RuntimeError(f"Depth round-trip failed for {path}")
    return int(depth.shape[0]), int(depth.shape[1])


def _matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "translation": arr[:3, 3].tolist(),
        "finite": bool(np.isfinite(arr).all()),
    }


def _intrinsic_summary(matrix: np.ndarray) -> dict[str, Any]:
    k = np.asarray(matrix, dtype=np.float64)
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
    }


def materialize_converted_gaussian_data(
    *,
    converted_session_path: Path,
    window_yaml: Path,
    camera_serials: list[str],
    anchor_serial: str,
    out_root: Path,
    case_name: str,
    shape_prior_glb: Path,
    world_transform: str = WORLD_TRANSFORM_PHYS_TWIN_Z_UP,
    frame_index: int = 0,
    overwrite: bool = False,
) -> Path:
    serials = _validate_camera_serials(camera_serials)
    if not shape_prior_glb.is_file() or shape_prior_glb.stat().st_size <= 0:
        raise FileNotFoundError(
            f"shape prior GLB must exist and be nonzero: {shape_prior_glb}"
        )

    case_dir = out_root / case_name
    _prepare_case_dir(case_dir, overwrite)

    backend = create_io_backend(
        io_backend="converted_session",
        converted_session_path=converted_session_path,
        camera_serials=serials,
        anchor_serial=anchor_serial,
        window_yaml=window_yaml,
        world_transform=world_transform,
    )

    cam_ids = backend.camera_ids()
    if cam_ids != list(range(len(serials))):
        raise RuntimeError(
            f"Unexpected backend camera ids {cam_ids}; expected 0..{len(serials) - 1}"
        )
    if len(cam_ids) != QQTT_CAMERA_COUNT:
        raise RuntimeError(
            f"Backend exposed {len(cam_ids)} cameras; expected {QQTT_CAMERA_COUNT}"
        )
    if frame_index < 0 or frame_index >= backend.frame_count():
        raise IndexError(
            f"frame_index {frame_index} out of range [0, {backend.frame_count()})"
        )

    image_files: dict[str, str] = {}
    depth_files: dict[str, str] = {}
    image_shapes: dict[str, list[int]] = {}
    depth_shapes: dict[str, list[int]] = {}
    source_frame_indices: dict[str, int | None] = {}
    width: int | None = None
    height: int | None = None

    for cam_id in cam_ids:
        rgb = backend.get_rgb(cam_id, frame_index)
        png_path = case_dir / f"{cam_id}.png"
        h, w = _write_rgb_png(rgb, png_path)
        if width is None:
            width, height = w, h
        elif (width, height) != (w, h):
            raise RuntimeError(
                f"Camera {cam_id} RGB resolution {(w, h)} differs from "
                f"camera 0 resolution {(width, height)}"
            )
        image_files[str(cam_id)] = png_path.name
        image_shapes[str(cam_id)] = [h, w, 3]

        depth_mm = backend.get_depth_mm(cam_id, frame_index)
        depth_path = case_dir / f"{cam_id}_depth.npy"
        dh, dw = _write_depth_mm(depth_mm, depth_path)
        if (dh, dw) != (h, w):
            raise RuntimeError(
                f"Camera {cam_id} depth resolution {(dh, dw)} differs from RGB {(h, w)}"
            )
        depth_files[str(cam_id)] = depth_path.name
        depth_shapes[str(cam_id)] = [dh, dw]

        source_frame_indices[str(cam_id)] = backend.source_frame_index(
            cam_id, frame_index
        )

    assert width is not None and height is not None

    c2ws = [np.asarray(backend.get_c2w(cam_id), dtype=np.float64) for cam_id in cam_ids]
    intrinsics = [
        np.asarray(backend.get_intrinsics(cam_id), dtype=np.float64) for cam_id in cam_ids
    ]
    for cam_id, c2w in enumerate(c2ws):
        if not np.isfinite(c2w).all():
            raise ValueError(f"Non-finite c2w for camera {cam_id}")
    for cam_id, k in enumerate(intrinsics):
        if k.shape != (3, 3):
            raise ValueError(f"Camera {cam_id} intrinsics shape {k.shape}, expected (3, 3)")

    camera_meta = {
        "c2ws": c2ws,
        "intrinsics": intrinsics,
        "width": int(width),
        "height": int(height),
    }
    camera_meta_path = case_dir / "camera_meta.pkl"
    with open(camera_meta_path, "wb") as f:
        pickle.dump(camera_meta, f)

    shape_prior_dst = case_dir / "shape_prior.glb"
    shutil.copy2(shape_prior_glb, shape_prior_dst)
    if shape_prior_dst.stat().st_size <= 0:
        raise RuntimeError(f"Copied shape prior is empty: {shape_prior_dst}")

    timestamp_ns: int | None = None
    try:
        timestamp_ns = int(backend.timestamp_ns(frame_index))
    except Exception:
        timestamp_ns = None

    repo_root = Path(__file__).resolve().parents[1]
    summary = {
        "script_version": SCRIPT_VERSION,
        "phystwin_git_head": _best_effort_git_head(repo_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "converted_session_path": str(converted_session_path.resolve()),
        "window_yaml": str(window_yaml.resolve()),
        "camera_serials": serials,
        "anchor_serial": anchor_serial,
        "world_transform": world_transform,
        "frame_index": int(frame_index),
        "backend_frame_count": int(backend.frame_count()),
        "timestamp_ns": timestamp_ns,
        "source_frame_indices": source_frame_indices,
        "output_case_path": str(case_dir.resolve()),
        "image_width": int(width),
        "image_height": int(height),
        "image_files": image_files,
        "image_shapes": image_shapes,
        "depth_files": depth_files,
        "depth_shapes": depth_shapes,
        "intrinsics_summary": [_intrinsic_summary(k) for k in intrinsics],
        "c2w_summary": [_matrix_summary(c2w) for c2w in c2ws],
        "shape_prior_source": str(shape_prior_glb.resolve()),
        "shape_prior_bytes": int(shape_prior_dst.stat().st_size),
        "camera_meta_path": camera_meta_path.name,
    }
    summary_path = case_dir / "materialization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"[materialize] wrote case: {case_dir}")
    print(f"[materialize] resolution: {width}x{height}, cameras: {len(cam_ids)}")
    return case_dir


def main() -> int:
    args = _parse_args()
    try:
        materialize_converted_gaussian_data(
            converted_session_path=args.converted_session_path,
            window_yaml=args.window_yaml,
            camera_serials=args.camera_serials,
            anchor_serial=args.anchor_serial,
            out_root=args.out_root,
            case_name=args.case_name,
            shape_prior_glb=args.shape_prior_glb,
            world_transform=args.world_transform,
            frame_index=args.frame_index,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[materialize] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
