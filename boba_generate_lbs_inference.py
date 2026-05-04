import argparse
import os
import pickle
from typing import List

import numpy as np
import torch

from gaussian_splatting.dynamic_utils import (
    get_topk_indices,
    knn_weights_sparse,
    calc_weights_vals_from_indices,
    interpolate_motions_speedup,
)


def get_setting_dirs(boba_root: str, settings: List[str] | None) -> List[str]:
    if settings is not None and len(settings) > 0:
        return [os.path.join(boba_root, setting) for setting in settings]

    all_dirs = []
    for name in sorted(os.listdir(boba_root)):
        path = os.path.join(boba_root, name)
        if os.path.isdir(path):
            all_dirs.append(path)
    return all_dirs


def lbs_reconstruct_dense_trajectory(
    bone_traj: np.ndarray,
    dense_first_frame: np.ndarray,
    k_weights: int = 16,
    k_rel: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    bone_traj_torch = torch.as_tensor(bone_traj, dtype=torch.float32, device=device)
    dense_first_torch = torch.as_tensor(dense_first_frame, dtype=torch.float32, device=device)

    num_frames, num_bones, _ = bone_traj_torch.shape
    dense_traj = torch.zeros(
        (num_frames, dense_first_torch.shape[0], 3),
        dtype=torch.float32,
        device=device,
    )
    dense_traj[0] = dense_first_torch

    rel_k = max(1, min(k_rel, num_bones - 1))
    w_k = max(1, min(k_weights, num_bones))

    relations = get_topk_indices(bone_traj_torch[0], K=rel_k)
    _, weights_indices = knn_weights_sparse(bone_traj_torch[0], dense_first_torch, K=w_k)

    for frame_idx in range(1, num_frames):
        prev_bones = bone_traj_torch[frame_idx - 1]
        curr_bones = bone_traj_torch[frame_idx]
        prev_dense = dense_traj[frame_idx - 1]

        weights = calc_weights_vals_from_indices(prev_bones, prev_dense, weights_indices)

        curr_dense, _, _ = interpolate_motions_speedup(
            bones=prev_bones,
            motions=curr_bones - prev_bones,
            relations=relations,
            weights=weights,
            weights_indices=weights_indices,
            xyz=prev_dense,
            quat=None,
            device=device,
            step=str(frame_idx),
        )
        dense_traj[frame_idx] = curr_dense

    return dense_traj.detach().cpu().numpy()


def process_case(
    case_dir: str,
    base_path: str,
    input_name: str,
    output_name: str,
    k_weights: int,
    k_rel: int,
    device: str,
) -> bool:
    case_name = os.path.basename(case_dir)
    input_path = os.path.join(case_dir, input_name)
    data_path = os.path.join(base_path, case_name, "final_data.pkl")
    output_path = os.path.join(case_dir, output_name)

    if not os.path.exists(input_path):
        print(f"[Skip] {case_name}: missing {input_path}")
        return False
    if not os.path.exists(data_path):
        print(f"[Skip] {case_name}: missing {data_path}")
        return False

    with open(input_path, "rb") as f:
        bone_traj = pickle.load(f)
    with open(data_path, "rb") as f:
        final_data = pickle.load(f)

    bone_traj = np.asarray(bone_traj, dtype=np.float32)
    object_points = np.asarray(final_data["object_points"], dtype=np.float32)

    if bone_traj.ndim != 3 or bone_traj.shape[-1] != 3:
        print(f"[Skip] {case_name}: invalid bone traj shape {bone_traj.shape}")
        return False
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        print(f"[Skip] {case_name}: invalid object_points shape {object_points.shape}")
        return False

    frame_len = min(bone_traj.shape[0], object_points.shape[0])
    if frame_len < 2:
        print(f"[Skip] {case_name}: frame_len={frame_len} too short")
        return False

    if bone_traj.shape[0] != object_points.shape[0]:
        print(
            f"[Warn] {case_name}: bone frames {bone_traj.shape[0]} != GT frames {object_points.shape[0]}, use first {frame_len}"
        )

    dense_traj = lbs_reconstruct_dense_trajectory(
        bone_traj=bone_traj[:frame_len],
        dense_first_frame=object_points[0],
        k_weights=k_weights,
        k_rel=k_rel,
        device=device,
    )

    with open(output_path, "wb") as f:
        pickle.dump(dense_traj.astype(np.float32), f)

    print(f"[Done] {case_name}: saved {output_path}, shape={dense_traj.shape}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boba_root", type=str, default="./boba_results")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--input_name", type=str, default="inference.pkl")
    parser.add_argument("--output_name", type=str, default="inference_lbs.pkl")
    parser.add_argument("--k_weights", type=int, default=16)
    parser.add_argument("--k_rel", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available, fallback to CPU")
        device = "cpu"
    else:
        device = args.device

    setting_dirs = get_setting_dirs(args.boba_root, args.settings)
    if len(setting_dirs) == 0:
        raise RuntimeError(f"No setting folders found in {args.boba_root}")

    total, success = 0, 0
    for setting_dir in setting_dirs:
        setting_name = os.path.basename(setting_dir)
        print(f"\n==== Processing setting: {setting_name} ====")

        case_names = sorted(
            name
            for name in os.listdir(setting_dir)
            if os.path.isdir(os.path.join(setting_dir, name))
        )

        for case_name in case_names:
            total += 1
            case_dir = os.path.join(setting_dir, case_name)
            ok = process_case(
                case_dir=case_dir,
                base_path=args.base_path,
                input_name=args.input_name,
                output_name=args.output_name,
                k_weights=args.k_weights,
                k_rel=args.k_rel,
                device=device,
            )
            success += int(ok)

    print(f"\nFinished: {success}/{total} cases processed successfully.")


if __name__ == "__main__":
    main()
