from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg, visualize_pc
from qqtt.utils.output_dirs import (
    add_experiments_dir_arg,
    add_experiments_optimization_dir_arg,
    add_reference_experiments_dir_arg,
    add_reference_experiments_optimization_dir_arg,
    experiments_case_dir,
    optimal_params_path,
    reference_experiments_root,
)
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json
import matplotlib.pyplot as plt


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_visualization_cfg(base_path: str, case_name: str) -> None:
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def _load_final_data_arrays(final_data_path: str):
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    object_points = data["object_points"]
    object_colors = data["object_colors"]
    object_visibilities = data["object_visibilities"]
    object_motions_valid = data["object_motions_valid"]
    controller_points = data["controller_points"]

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]
    rainbow_colors = np.tile(rainbow_colors[None, :, :], (object_points.shape[0], 1, 1))
    return (
        object_points,
        rainbow_colors,
        controller_points,
        object_visibilities,
        object_motions_valid,
    )


def render_gt_video_from_final_data(
    *,
    base_path: str,
    case_name: str,
    output_path: str,
    final_data_path: str | None = None,
) -> None:
    final_data_path = final_data_path or f"{base_path}/{case_name}/final_data.pkl"
    load_visualization_cfg(base_path, case_name)
    cfg.disable_video_logging = False
    (
        object_points,
        object_colors,
        controller_points,
        object_visibilities,
        object_motions_valid,
    ) = _load_final_data_arrays(final_data_path)
    visualize_pc(
        object_points,
        object_colors,
        controller_points,
        object_visibilities,
        object_motions_valid,
        visualize=False,
        save_video=True,
        save_path=output_path,
    )


def render_inference_video_from_pkl(
    *,
    base_path: str,
    case_name: str,
    inference_pkl: str,
    output_path: str,
    final_data_path: str | None = None,
) -> None:
    final_data_path = final_data_path or f"{base_path}/{case_name}/final_data.pkl"
    load_visualization_cfg(base_path, case_name)
    cfg.disable_video_logging = False
    with open(inference_pkl, "rb") as f:
        vertices = pickle.load(f)
    (
        _gt_points,
        object_colors,
        controller_points,
        object_visibilities,
        object_motions_valid,
    ) = _load_final_data_arrays(final_data_path)
    num_all = vertices.shape[1]
    if object_colors.shape[1] > num_all:
        object_colors = object_colors[:, :num_all, :]
    elif object_colors.shape[1] < num_all:
        pad = np.ones((object_colors.shape[0], num_all - object_colors.shape[1], 3)) * 0.3
        object_colors = np.concatenate([object_colors, pad], axis=1)
    # Match trainer_warp.py pure-inference rendering: do not pass GT visibility
    # masks onto the full simulated trajectory (e.g. 6625 tracked vs 9864 simulated).
    visualize_pc(
        vertices,
        object_colors,
        controller_points,
        None,
        None,
        visualize=False,
        save_video=True,
        save_path=output_path,
    )


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    add_experiments_optimization_dir_arg(parser)
    add_reference_experiments_optimization_dir_arg(parser)
    add_experiments_dir_arg(parser)
    add_reference_experiments_dir_arg(parser)
    parser.add_argument(
        "--render-gt-video-only",
        action="store_true",
        help="Render GT trajectory video from existing final_data.pkl (no inference).",
    )
    parser.add_argument(
        "--render-inference-video-only",
        action="store_true",
        help="Render inference.mp4 from an existing inference.pkl (no simulation).",
    )
    parser.add_argument(
        "--inference-pkl",
        type=str,
        default=None,
        help="Path to inference.pkl for --render-inference-video-only.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Explicit MP4 output path for render-only modes.",
    )
    parser.add_argument(
        "--final-data-pkl",
        type=str,
        default=None,
        help="Optional final_data.pkl override for render-only modes.",
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    if args.render_gt_video_only:
        output_path = args.output_video or f"{base_path}/{case_name}/gt.mp4"
        render_gt_video_from_final_data(
            base_path=base_path,
            case_name=case_name,
            output_path=output_path,
            final_data_path=args.final_data_pkl,
        )
        raise SystemExit(0)

    if args.render_inference_video_only:
        inference_pkl = args.inference_pkl
        if inference_pkl is None:
            base_dir = experiments_case_dir(args, case_name)
            inference_pkl = f"{base_dir}/inference.pkl"
        output_path = args.output_video or f"{experiments_case_dir(args, case_name)}/inference.mp4"
        if not os.path.isfile(inference_pkl):
            raise FileNotFoundError(f"inference.pkl not found: {inference_pkl}")
        render_inference_video_from_pkl(
            base_path=base_path,
            case_name=case_name,
            inference_pkl=inference_pkl,
            output_path=output_path,
            final_data_path=args.final_data_pkl,
        )
        raise SystemExit(0)

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = experiments_case_dir(args, case_name)

    optimal_path = optimal_params_path(args, case_name)
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    load_visualization_cfg(base_path, case_name)

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    checkpoint_root = reference_experiments_root(args)
    checkpoint_dir = os.path.join(checkpoint_root, case_name, "train")
    assert len(glob.glob(f"{checkpoint_dir}/best_*.pth")) > 0
    best_model_path = glob.glob(f"{checkpoint_dir}/best_*.pth")[0]
    trainer.test(best_model_path)
