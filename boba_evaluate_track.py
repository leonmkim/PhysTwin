import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
from scipy.spatial import KDTree


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0.0
        else:
            track_error = float(np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1)))
        track_errors.append(track_error)

    return float(np.mean(track_errors)) if len(track_errors) > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boba_root", type=str, default="./boba_results")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--settings", nargs="*", default=["Boba", "Boba_Local"])
    parser.add_argument("--inference_name", type=str, default="inference_lbs.pkl")
    parser.add_argument("--output_file", type=str, default="results/final_track_boba_lbs.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Setting", "Case Name", "Train Track Error", "Test Track Error"])

        for setting in args.settings:
            prediction_path = os.path.join(args.boba_root, setting)
            dir_names = sorted(glob.glob(f"{prediction_path}/*"))

            for dir_name in dir_names:
                case_name = os.path.basename(dir_name)
                split_path = os.path.join(args.base_path, case_name, "split.json")
                inference_path = os.path.join(dir_name, args.inference_name)
                gt_track_path = os.path.join(args.base_path, case_name, "gt_track_3d.pkl")

                if not os.path.exists(split_path):
                    print(f"[Skip] {setting}/{case_name}: missing split")
                    continue
                if not os.path.exists(inference_path):
                    print(f"[Skip] {setting}/{case_name}: missing {inference_path}")
                    continue
                if not os.path.exists(gt_track_path):
                    print(f"[Skip] {setting}/{case_name}: missing gt_track_3d.pkl")
                    continue

                print(f"Processing {setting}/{case_name}")

                with open(split_path, "r") as f:
                    split = json.load(f)
                train_frame = split["train"][1]
                test_frame = split["test"][1]

                with open(inference_path, "rb") as f:
                    vertices = np.asarray(pickle.load(f), dtype=np.float32)

                with open(gt_track_path, "rb") as f:
                    gt_track_3d = np.asarray(pickle.load(f), dtype=np.float32)

                if vertices.shape[0] < test_frame:
                    print(
                        f"[Warn] {setting}/{case_name}: vertices frames {vertices.shape[0]} < test_frame {test_frame}, clip split"
                    )
                    test_frame = vertices.shape[0]
                    train_frame = min(train_frame, test_frame)

                mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
                kdtree = KDTree(vertices[0])
                _, idx = kdtree.query(gt_track_3d[0][mask])

                train_track_error = evaluate_prediction(
                    1, train_frame, vertices, gt_track_3d, idx, mask
                )
                test_track_error = evaluate_prediction(
                    train_frame, test_frame, vertices, gt_track_3d, idx, mask
                )

                writer.writerow([setting, case_name, train_track_error, test_track_error])

    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()
