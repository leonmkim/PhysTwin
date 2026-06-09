"""CLI helpers for output-root overrides (Stage C scratch safety)."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
import os

DEFAULT_EXPERIMENTS_OPTIMIZATION_DIR = "experiments_optimization"
DEFAULT_EXPERIMENTS_DIR = "experiments"
DEFAULT_GAUSSIAN_OUTPUT_DIR = "gaussian_output"
DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_DIR = "gaussian_output_dynamic"
DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR = "gaussian_output_dynamic_white"

SCRATCH_EXPERIMENTS_OPTIMIZATION_DIR = "temp_experiments_optimization_uv"
SCRATCH_EXPERIMENTS_DIR = "temp_experiments_uv"
SCRATCH_GAUSSIAN_OUTPUT_DIR = "temp_gaussian_output_uv"
SCRATCH_GAUSSIAN_OUTPUT_DYNAMIC_DIR = "temp_gaussian_output_dynamic_uv"
SCRATCH_GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR = "temp_gaussian_output_dynamic_white_uv"


def add_experiments_optimization_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--experiments-optimization-dir",
        default=DEFAULT_EXPERIMENTS_OPTIMIZATION_DIR,
        help=(
            "Directory for CMA optimization outputs "
            f"(default: {DEFAULT_EXPERIMENTS_OPTIMIZATION_DIR})"
        ),
    )


def add_reference_experiments_optimization_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--reference-experiments-optimization-dir",
        default=None,
        help=(
            "Read optimal_params.pkl from this root "
            "(default: --experiments-optimization-dir)"
        ),
    )


def add_experiments_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--experiments-dir",
        default=DEFAULT_EXPERIMENTS_DIR,
        help=f"Directory for warp train/inference outputs (default: {DEFAULT_EXPERIMENTS_DIR})",
    )


def add_reference_experiments_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--reference-experiments-dir",
        default=None,
        help="Read checkpoints/inference.pkl from this root (default: --experiments-dir)",
    )


def add_gaussian_output_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--gaussian-output-dir",
        default=DEFAULT_GAUSSIAN_OUTPUT_DIR,
        help=f"Directory for Gaussian splatting training output (default: {DEFAULT_GAUSSIAN_OUTPUT_DIR})",
    )


def add_reference_gaussian_output_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--reference-gaussian-output-dir",
        default=None,
        help="Read trained Gaussian models from this root (default: --gaussian-output-dir)",
    )


def add_gaussian_output_dynamic_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--gaussian-output-dynamic-dir",
        default=DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_DIR,
        help=(
            "Directory for dynamic Gaussian render output "
            f"(default: {DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_DIR})"
        ),
    )


def add_gaussian_output_dynamic_white_dir_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--gaussian-output-dynamic-white-dir",
        default=DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR,
        help=(
            "Directory for white-background dynamic render output "
            f"(default: {DEFAULT_GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR})"
        ),
    )


def reference_experiments_optimization_root(args: Namespace) -> str:
    ref = getattr(args, "reference_experiments_optimization_dir", None)
    if ref:
        return ref
    return args.experiments_optimization_dir


def reference_experiments_root(args: Namespace) -> str:
    ref = getattr(args, "reference_experiments_dir", None)
    if ref:
        return ref
    return args.experiments_dir


def reference_gaussian_output_root(args: Namespace) -> str:
    ref = getattr(args, "reference_gaussian_output_dir", None)
    if ref:
        return ref
    return args.gaussian_output_dir


def experiments_optimization_case_dir(args: Namespace, case_name: str) -> str:
    return os.path.join(args.experiments_optimization_dir, case_name)


def experiments_case_dir(args: Namespace, case_name: str) -> str:
    return os.path.join(args.experiments_dir, case_name)


def optimal_params_path(args: Namespace, case_name: str) -> str:
    return os.path.join(
        reference_experiments_optimization_root(args), case_name, "optimal_params.pkl"
    )


def inference_pkl_path(args: Namespace, case_name: str, *, reference: bool = True) -> str:
    root = reference_experiments_root(args) if reference else args.experiments_dir
    return os.path.join(root, case_name, "inference.pkl")
