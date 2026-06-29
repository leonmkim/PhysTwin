"""Materialize native QQTT Gaussian data with all-camera masks from a converted session.

Extends :mod:`materialize_converted_gaussian_data` with cam0 mask reuse,
GroundedSAM masks for cameras 1/2, validation summaries, overlays, and an
optional shape-prior projection audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from data_process.io_backend import WORLD_TRANSFORM_PHYS_TWIN_Z_UP
from data_process.materialize_converted_gaussian_data import (
    MIN_QQTT_CAMERA_COUNT,
    materialize_converted_gaussian_data,
)

SCRIPT_VERSION = "1.0.0"

MIN_MASK_AREA_FRACTION = 0.001
MAX_MASK_AREA_FRACTION = 0.60
NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize native QQTT gaussian_data with object/occlusion masks "
            "from a converted session."
        ),
    )
    parser.add_argument("--converted-session-path", type=Path, required=True)
    parser.add_argument("--window-yaml", type=Path, required=True)
    parser.add_argument("--camera-serials", nargs="+", required=True)
    parser.add_argument("--anchor-serial", type=str, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--case-name", type=str, required=True)
    parser.add_argument("--shape-prior-glb", type=Path, required=True)
    parser.add_argument(
        "--world-transform",
        type=str,
        default=WORLD_TRANSFORM_PHYS_TWIN_Z_UP,
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--cam0-object-mask", type=Path, required=True)
    parser.add_argument("--cam0-occlusion-mask", type=Path, required=True)
    parser.add_argument("--object-prompt", type=str, default="plush")
    parser.add_argument("--occlusion-prompt", type=str, default="human")
    parser.add_argument(
        "--occlusion-fallback-prompts",
        nargs="*",
        default=["hand", "arm", "hand.arm"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--projection-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help="Python for segment_util_image.py (default: sys.executable).",
    )
    parser.add_argument(
        "--segment-script",
        type=Path,
        default=None,
        help="Path to segment_util_image.py (default: repo data_process/).",
    )
    return parser.parse_args()


def load_mask_alpha(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unreadable mask: {path}")
    if img.ndim == 2:
        return img.astype(np.float32) / 255.0
    if img.shape[2] == 4:
        return img[:, :, 3].astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 127).astype(np.float32)


def mask_stats(
    alpha: np.ndarray,
    *,
    min_area: float = MIN_MASK_AREA_FRACTION,
    max_area: float = MAX_MASK_AREA_FRACTION,
) -> dict[str, Any]:
    h, w = alpha.shape[:2]
    fg = alpha > 0.5
    area_frac = float(fg.mean())
    ys, xs = np.where(fg)
    centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else None
    return {
        "shape_hw": [int(h), int(w)],
        "area_fraction": area_frac,
        "centroid_xy": centroid,
        "nonempty": area_frac > min_area,
        "not_full_frame": area_frac < max_area,
        "valid": area_frac > min_area and area_frac < max_area,
    }


def validate_mask_file(
    path: Path,
    *,
    expected_hw: tuple[int, int] = (NATIVE_HEIGHT, NATIVE_WIDTH),
) -> dict[str, Any]:
    alpha = load_mask_alpha(path)
    stats = mask_stats(alpha)
    stats["path"] = str(path)
    if tuple(stats["shape_hw"]) != expected_hw:
        raise ValueError(
            f"Mask {path} shape {stats['shape_hw']} != expected {list(expected_hw)}"
        )
    if not stats["valid"]:
        raise ValueError(
            f"Mask {path} failed validation: area_fraction={stats['area_fraction']:.4f}"
        )
    return stats


def mask_overlap_fraction(alpha_a: np.ndarray, alpha_b: np.ndarray) -> float:
    a = alpha_a > 0.5
    b = alpha_b > 0.5
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def is_usable_mask_stats(stats: dict[str, Any]) -> bool:
    return bool(stats.get("valid"))


def select_occlusion_prompt_results(
    prompt_results: list[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    """Pick the first usable occlusion mask prompt result."""
    for prompt, stats in prompt_results:
        if is_usable_mask_stats(stats):
            return prompt, stats
    prompts = [p for p, _ in prompt_results]
    areas = [s.get("area_fraction") for _, s in prompt_results]
    raise RuntimeError(
        "No usable occlusion mask after prompts "
        f"{prompts}; area_fractions={areas}"
    )


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_mask(src: Path, dst: Path) -> dict[str, Any]:
    if not src.is_file() or src.stat().st_size <= 0:
        raise FileNotFoundError(f"Mask source missing or empty: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)
    stats = validate_mask_file(dst)
    stats["source"] = str(src.resolve())
    stats["md5"] = md5_file(dst)
    return stats


def make_overlay(rgb_path: Path, alpha: np.ndarray, out_path: Path) -> None:
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(f"Unreadable RGB: {rgb_path}")
    overlay = rgb.copy()
    mask = alpha > 0.5
    overlay[mask] = (0.35 * overlay[mask] + 0.65 * np.array([0, 0, 255])).astype(
        np.uint8
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), overlay):
        raise RuntimeError(f"Failed to write overlay: {out_path}")


def run_grounded_sam_segmentation(
    *,
    repo_root: Path,
    python_executable: Path,
    segment_script: Path,
    img_path: Path,
    output_path: Path,
    text_prompt: str,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(extra_env or {})
    cmd = [
        str(python_executable),
        str(segment_script),
        "--img_path",
        str(img_path),
        "--TEXT_PROMPT",
        text_prompt,
        "--output_path",
        str(output_path),
    ]
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"command: {' '.join(cmd)}\n")
        log_f.write(f"cwd: {repo_root}\n\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"segment_util_image.py failed for prompt '{text_prompt}' "
            f"(exit {proc.returncode}); log={log_path}"
        )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"segment_util_image.py produced no output for prompt '{text_prompt}'; "
            f"log={log_path}"
        )


def segment_mask_with_prompt(
    *,
    repo_root: Path,
    python_executable: Path,
    segment_script: Path,
    cam_id: int,
    kind: str,
    img_path: Path,
    output_path: Path,
    prompt: str,
    log_dir: Path,
    extra_env: dict[str, str] | None,
) -> dict[str, Any]:
    log_path = log_dir / f"segment_{kind}_cam{cam_id}_{prompt.replace('.', '_')}.log"
    if output_path.exists():
        output_path.unlink()
    run_grounded_sam_segmentation(
        repo_root=repo_root,
        python_executable=python_executable,
        segment_script=segment_script,
        img_path=img_path,
        output_path=output_path,
        text_prompt=prompt,
        log_path=log_path,
        extra_env=extra_env,
    )
    stats = validate_mask_file(output_path)
    stats["prompt"] = prompt
    stats["log_path"] = str(log_path)
    return stats


def project_shape_prior_audit(
    *,
    camera_meta_path: Path,
    shape_prior_path: Path,
    segmented_mask_paths: dict[int, Path],
    width: int = NATIVE_WIDTH,
    height: int = NATIVE_HEIGHT,
    max_vertices: int = 20000,
) -> dict[str, Any]:
    with open(camera_meta_path, "rb") as f:
        meta = pickle.load(f)
    c2ws = meta["c2ws"]
    intrinsics = meta["intrinsics"]

    mesh = trimesh.load_mesh(str(shape_prior_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) > max_vertices:
        rng = np.random.default_rng(0)
        verts = verts[rng.choice(len(verts), max_vertices, replace=False)]

    def project_points(
        pts_world: np.ndarray, c2w: np.ndarray, k: np.ndarray
    ) -> np.ndarray:
        w2c = np.linalg.inv(c2w)
        pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
        cam = (w2c @ pts_h.T).T[:, :3]
        z = cam[:, 2]
        valid = z > 1e-4
        cam = cam[valid]
        u = k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]
        v = k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]
        uv = np.stack([u, v], axis=1)
        good = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        return uv[good]

    def rasterize_points(uv: np.ndarray) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in uv.astype(int):
            cv2.circle(mask, (int(x), int(y)), 3, 1, -1)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask.astype(bool)

    def centroid(mask: np.ndarray) -> list[float] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [float(xs.mean()), float(ys.mean())]

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    out: dict[str, Any] = {
        "method": "sampled_mesh_vertex_projection_with_dilation",
        "limitations": [
            "Point projection + dilation, not full mesh rasterization",
        ],
        "cameras": {},
        "flags": [],
    }

    for cam_id, mask_path in segmented_mask_paths.items():
        if cam_id == 0:
            continue
        uv = project_points(
            verts, np.asarray(c2ws[cam_id]), np.asarray(intrinsics[cam_id])
        )
        proj_mask = rasterize_points(uv)
        seg_alpha = load_mask_alpha(mask_path)
        seg_mask = seg_alpha > 0.5
        proj_c = centroid(proj_mask)
        seg_c = centroid(seg_mask)
        offset = None
        offset_px = None
        if proj_c and seg_c:
            offset = [seg_c[0] - proj_c[0], seg_c[1] - proj_c[1]]
            offset_px = float(np.hypot(offset[0], offset[1]))
        iou_val = iou(proj_mask, seg_mask)
        cam_out = {
            "segmented_mask": str(mask_path),
            "projected_area_fraction": float(proj_mask.mean()),
            "segmented_area_fraction": float(seg_mask.mean()),
            "projected_centroid_xy": proj_c,
            "segmented_centroid_xy": seg_c,
            "centroid_offset_xy": offset,
            "centroid_offset_px": offset_px,
            "approx_iou": iou_val,
        }
        out["cameras"][str(cam_id)] = cam_out
        if iou_val < 0.3:
            out["flags"].append(f"cam{cam_id}: IoU {iou_val:.3f} < 0.3")
        if offset_px is not None and offset_px > 200:
            out["flags"].append(
                f"cam{cam_id}: centroid offset {offset_px:.1f}px > 200"
            )
    return out


def materialize_converted_masked_gaussian_data(
    *,
    converted_session_path: Path,
    window_yaml: Path,
    camera_serials: list[str],
    anchor_serial: str,
    out_root: Path,
    case_name: str,
    shape_prior_glb: Path,
    cam0_object_mask: Path,
    cam0_occlusion_mask: Path,
    world_transform: str = WORLD_TRANSFORM_PHYS_TWIN_Z_UP,
    frame_index: int = 0,
    object_prompt: str = "plush",
    occlusion_prompt: str = "human",
    occlusion_fallback_prompts: list[str] | None = None,
    overwrite: bool = False,
    write_overlays: bool = True,
    projection_audit: bool = True,
    python_executable: Path | None = None,
    segment_script: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    if occlusion_fallback_prompts is None:
        occlusion_fallback_prompts = ["hand", "arm", "hand.arm"]

    camera_count = len([s for s in camera_serials if str(s).strip()])
    if camera_count < MIN_QQTT_CAMERA_COUNT:
        raise ValueError(
            f"Masked Gaussian materialization requires at least "
            f"{MIN_QQTT_CAMERA_COUNT} cameras; got {camera_count}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    py = python_executable or Path(sys.executable)
    seg_script = segment_script or (repo_root / "data_process" / "segment_util_image.py")
    if not seg_script.is_file():
        raise FileNotFoundError(f"segment_util_image.py not found: {seg_script}")

    case_dir = materialize_converted_gaussian_data(
        converted_session_path=converted_session_path,
        window_yaml=window_yaml,
        camera_serials=camera_serials,
        anchor_serial=anchor_serial,
        out_root=out_root,
        case_name=case_name,
        shape_prior_glb=shape_prior_glb,
        world_transform=world_transform,
        frame_index=frame_index,
        overwrite=overwrite,
    )

    log_dir = case_dir / "segmentation_logs"
    mask_summary: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(case_dir.resolve()),
        "cameras": {},
        "object_prompt_default": object_prompt,
        "occlusion_prompt_default": occlusion_prompt,
        "occlusion_fallback_prompts": list(occlusion_fallback_prompts),
    }

    cam0_obj_dst = case_dir / "mask_0.png"
    cam0_occ_dst = case_dir / "mask_human_0.png"
    mask_summary["cameras"]["0"] = {
        "object": {
            **copy_mask(cam0_object_mask, cam0_obj_dst),
            "source_kind": "copy",
        },
        "occlusion": {
            **copy_mask(cam0_occlusion_mask, cam0_occ_dst),
            "source_kind": "copy",
        },
    }

    for cam_id in range(1, camera_count):
        img_path = case_dir / f"{cam_id}.png"
        obj_path = case_dir / f"mask_{cam_id}.png"
        occ_path = case_dir / f"mask_human_{cam_id}.png"

        obj_stats = segment_mask_with_prompt(
            repo_root=repo_root,
            python_executable=py,
            segment_script=seg_script,
            cam_id=cam_id,
            kind="object",
            img_path=img_path,
            output_path=obj_path,
            prompt=object_prompt,
            log_dir=log_dir,
            extra_env=extra_env,
        )

        occ_prompt_results: list[tuple[str, dict[str, Any]]] = []
        occ_prompts = [occlusion_prompt, *occlusion_fallback_prompts]
        chosen_prompt = None
        chosen_stats = None
        for prompt in occ_prompts:
            try:
                stats = segment_mask_with_prompt(
                    repo_root=repo_root,
                    python_executable=py,
                    segment_script=seg_script,
                    cam_id=cam_id,
                    kind="occlusion",
                    img_path=img_path,
                    output_path=occ_path,
                    prompt=prompt,
                    log_dir=log_dir,
                    extra_env=extra_env,
                )
            except (RuntimeError, ValueError) as exc:
                occ_prompt_results.append((prompt, {"valid": False, "error": str(exc)}))
                continue
            occ_prompt_results.append((prompt, stats))
            if is_usable_mask_stats(stats):
                chosen_prompt = prompt
                chosen_stats = stats
                break
        if chosen_prompt is None or chosen_stats is None:
            chosen_prompt, chosen_stats = select_occlusion_prompt_results(
                occ_prompt_results
            )

        obj_alpha = load_mask_alpha(obj_path)
        occ_alpha = load_mask_alpha(occ_path)
        overlap = mask_overlap_fraction(obj_alpha, occ_alpha)

        mask_summary["cameras"][str(cam_id)] = {
            "object": {**obj_stats, "source_kind": "groundedsam"},
            "occlusion": {
                **chosen_stats,
                "source_kind": "groundedsam",
                "prompts_tried": [p for p, _ in occ_prompt_results],
            },
            "object_occlusion_overlap_fraction": overlap,
        }

    if write_overlays:
        overlay_dir = case_dir / "mask_overlays"
        for cam_id in range(camera_count):
            rgb_path = case_dir / f"{cam_id}.png"
            obj_alpha = load_mask_alpha(case_dir / f"mask_{cam_id}.png")
            occ_alpha = load_mask_alpha(case_dir / f"mask_human_{cam_id}.png")
            make_overlay(
                rgb_path, obj_alpha, overlay_dir / f"cam{cam_id}_object_overlay.png"
            )
            make_overlay(
                rgb_path,
                occ_alpha,
                overlay_dir / f"cam{cam_id}_occlusion_overlay.png",
            )
        mask_summary["overlay_dir"] = str(overlay_dir.resolve())

    mask_summary_path = case_dir / "mask_validation_summary.json"
    with open(mask_summary_path, "w", encoding="utf-8") as f:
        json.dump(mask_summary, f, indent=2)
        f.write("\n")

    if projection_audit:
        audit = project_shape_prior_audit(
            camera_meta_path=case_dir / "camera_meta.pkl",
            shape_prior_path=case_dir / "shape_prior.glb",
            segmented_mask_paths={
                cam_id: case_dir / f"mask_{cam_id}.png"
                for cam_id in range(camera_count)
            },
        )
        audit_path = case_dir / "projection_audit_summary.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
            f.write("\n")
        print(
            "[masked-materialize] projection audit flags:",
            audit.get("flags") or "none",
        )

    print(f"[masked-materialize] wrote masked case: {case_dir}")
    return case_dir


def main() -> int:
    args = _parse_args()
    extra_env = dict(os.environ)
    gsam_root = extra_env.get("GSAM_ROOT")
    if gsam_root:
        py_path = extra_env.get("PYTHONPATH", "")
        extra_env["PYTHONPATH"] = f"{gsam_root}:{py_path}" if py_path else gsam_root

    try:
        materialize_converted_masked_gaussian_data(
            converted_session_path=args.converted_session_path,
            window_yaml=args.window_yaml,
            camera_serials=args.camera_serials,
            anchor_serial=args.anchor_serial,
            out_root=args.out_root,
            case_name=args.case_name,
            shape_prior_glb=args.shape_prior_glb,
            cam0_object_mask=args.cam0_object_mask,
            cam0_occlusion_mask=args.cam0_occlusion_mask,
            world_transform=args.world_transform,
            frame_index=args.frame_index,
            object_prompt=args.object_prompt,
            occlusion_prompt=args.occlusion_prompt,
            occlusion_fallback_prompts=list(args.occlusion_fallback_prompts),
            overwrite=args.overwrite,
            write_overlays=args.write_overlays,
            projection_audit=args.projection_audit,
            python_executable=args.python_executable,
            segment_script=args.segment_script,
            extra_env=extra_env,
        )
    except Exception as exc:
        print(f"[masked-materialize] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
