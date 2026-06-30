# The first stage to optimize the sparse parameters using CMA-ES
from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.output_dirs import (
    add_experiments_optimization_dir_arg,
    experiments_optimization_case_dir,
)
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import torch
import pickle
import json
import os
import tempfile
from argparse import ArgumentParser


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
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def render_cma_rollout_video_only(
    *,
    base_path: str,
    case_name: str,
    train_frame: int,
    rollout_kind: str,
    output_video_path: str,
    optimal_params_path: str | None = None,
    scratch_base_dir: str | None = None,
) -> dict[str, object]:
    """Render one CMA rollout video without invoking CMA-ES."""
    if rollout_kind not in {"init", "optimal"}:
        raise ValueError(f"Unsupported rollout kind: {rollout_kind}")

    load_visualization_cfg(base_path, case_name)
    cfg.disable_video_logging = False

    if rollout_kind == "optimal":
        if optimal_params_path is None or not os.path.isfile(optimal_params_path):
            raise FileNotFoundError(
                f"optimal_params.pkl not found for optimal rollout: {optimal_params_path}"
            )

    scratch_dir = scratch_base_dir
    if scratch_dir is None:
        scratch_dir = tempfile.mkdtemp(prefix="phystwin_cma_rollout_")
    os.makedirs(scratch_dir, exist_ok=True)

    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=scratch_dir,
        train_frame=train_frame,
    )

    if rollout_kind == "init":
        parameters = optimizer.build_initial_cma_parameters()
    else:
        with open(optimal_params_path, "rb") as handle:
            optimal_results = pickle.load(handle)
        parameters = optimizer.optimal_results_to_cma_parameters(optimal_results)

    return optimizer.render_rollout_video(
        parameters,
        output_path=output_video_path,
        label=rollout_kind,
    )


seed = 42
set_all_seeds(seed)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument(
        "--disable-video-logging",
        action="store_true",
        help="Disable Open3D diagnostic video rendering (headless-safe).",
    )
    parser.add_argument(
        "--render-cma-rollout-video-only",
        action="store_true",
        help="Render one CMA rollout diagnostic video without running CMA-ES.",
    )
    parser.add_argument(
        "--cma-rollout-kind",
        choices=["init", "optimal"],
        default=None,
        help="Rollout parameter source for --render-cma-rollout-video-only.",
    )
    parser.add_argument(
        "--optimal-params-path",
        type=str,
        default=None,
        help="Path to optimal_params.pkl for --cma-rollout-kind optimal.",
    )
    parser.add_argument(
        "--output-video-path",
        type=str,
        default=None,
        help="Explicit MP4 output path for --render-cma-rollout-video-only.",
    )
    add_experiments_optimization_dir_arg(parser)
    args = parser.parse_args(argv)

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    if args.render_cma_rollout_video_only:
        if args.cma_rollout_kind is None:
            raise SystemExit("--cma-rollout-kind is required with --render-cma-rollout-video-only")
        if args.output_video_path is None:
            raise SystemExit("--output-video-path is required with --render-cma-rollout-video-only")
        optimal_path = args.optimal_params_path
        if args.cma_rollout_kind == "optimal" and optimal_path is None:
            optimal_path = os.path.join(
                experiments_optimization_case_dir(args, case_name),
                "optimal_params.pkl",
            )
        result = render_cma_rollout_video_only(
            base_path=base_path,
            case_name=case_name,
            train_frame=train_frame,
            rollout_kind=args.cma_rollout_kind,
            output_video_path=args.output_video_path,
            optimal_params_path=optimal_path,
        )
        logger.info(f"CMA rollout render complete: {result}")
        raise SystemExit(0)

    if args.disable_video_logging:
        cfg.disable_video_logging = True

    base_dir = experiments_optimization_case_dir(args, case_name)
    load_visualization_cfg(base_path, case_name)

    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    optimizer.optimize(max_iter=max_iter)


if __name__ == "__main__":
    main()
