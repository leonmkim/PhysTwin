import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance


def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        current_pred = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]

        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_pred = current_pred[:num_surface_points]

        chamfer_out = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_pred.unsqueeze(0),
            single_directional=True,
            norm=1,
        )
        if isinstance(chamfer_out, tuple):
            chamfer_value = chamfer_out[0]
            if isinstance(chamfer_value, tuple):
                chamfer_value = chamfer_value[0]
        else:
            chamfer_value = chamfer_out
        chamfer_errors.append(float(torch.as_tensor(chamfer_value).item()))

    chamfer_errors = np.array(chamfer_errors, dtype=np.float32)
    return {
        "frame_len": int(len(chamfer_errors)),
        "chamfer_error": float(np.mean(chamfer_errors)) if len(chamfer_errors) > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boba_root", type=str, default="./boba_results")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--settings", nargs="*", default=["Boba", "Boba_Local"])
    parser.add_argument("--inference_name", type=str, default="inference_lbs.pkl")
    parser.add_argument("--output_file", type=str, default="boba_results/final_results_boba_lbs.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Setting",
                "Case Name",
                "Train Frame Num",
                "Train Chamfer Error",
                "Test Frame Num",
                "Test Chamfer Error",
            ]
        )

        for setting in args.settings:
            prediction_dir = os.path.join(args.boba_root, setting)
            case_dirs = sorted(glob.glob(f"{prediction_dir}/*"))

            for dir_name in case_dirs:
                case_name = os.path.basename(dir_name)
                inference_path = os.path.join(dir_name, args.inference_name)
                gt_data_path = os.path.join(args.base_path, case_name, "final_data.pkl")
                split_path = os.path.join(args.base_path, case_name, "split.json")

                if not os.path.exists(inference_path):
                    print(f"[Skip] {setting}/{case_name}: missing {inference_path}")
                    continue
                if not os.path.exists(gt_data_path) or not os.path.exists(split_path):
                    print(f"[Skip] {setting}/{case_name}: missing GT data/split")
                    continue

                print(f"Processing {setting}/{case_name}")

                with open(inference_path, "rb") as f:
                    vertices = np.asarray(pickle.load(f), dtype=np.float32)

                with open(gt_data_path, "rb") as f:
                    data = pickle.load(f)

                object_points = np.asarray(data["object_points"], dtype=np.float32)
                object_visibilities = np.asarray(data["object_visibilities"], dtype=bool)
                num_original_points = object_points.shape[1]
                num_surface_points = num_original_points + np.asarray(data["surface_points"]).shape[0]

                with open(split_path, "r") as f:
                    split = json.load(f)
                train_frame = split["train"][1]
                test_frame = split["test"][1]

                if vertices.shape[0] < test_frame:
                    print(
                        f"[Warn] {setting}/{case_name}: vertices frames {vertices.shape[0]} < test_frame {test_frame}, clip split"
                    )
                    test_frame = vertices.shape[0]
                    train_frame = min(train_frame, test_frame)

                results_train = evaluate_prediction(
                    1,
                    train_frame,
                    vertices,
                    object_points,
                    object_visibilities,
                    num_surface_points,
                )
                results_test = evaluate_prediction(
                    train_frame,
                    test_frame,
                    vertices,
                    object_points,
                    object_visibilities,
                    num_surface_points,
                )

                writer.writerow(
                    [
                        setting,
                        case_name,
                        results_train["frame_len"],
                        results_train["chamfer_error"],
                        results_test["frame_len"],
                        results_test["chamfer_error"],
                    ]
                )

    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
