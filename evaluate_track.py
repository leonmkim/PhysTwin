import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
from scipy.spatial import KDTree

from qqtt.utils.output_dirs import add_experiments_dir_arg, add_reference_experiments_dir_arg


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))

        track_errors.append(track_error)
    return np.mean(track_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    add_experiments_dir_arg(parser)
    add_reference_experiments_dir_arg(parser)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for evaluation CSV output (default: results)",
    )
    args = parser.parse_args()

    prediction_path = (
        args.reference_experiments_dir
        if args.reference_experiments_dir
        else args.experiments_dir
    )
    base_path = args.base_path
    os.makedirs(args.results_dir, exist_ok=True)
    output_file = os.path.join(args.results_dir, "final_track.csv")

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "Case Name",
            "Train Track Error",
            "Test Track Error",
        ]
    )

    dir_names = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        with open(f"{prediction_path}/{case_name}/inference.pkl", "rb") as f:
            vertices = pickle.load(f)

        with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
            gt_track_3d = pickle.load(f)

        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

        kdtree = KDTree(vertices[0])
        dis, idx = kdtree.query(gt_track_3d[0][mask])

        train_track_error = evaluate_prediction(
            1, train_frame, vertices, gt_track_3d, idx, mask
        )
        test_track_error = evaluate_prediction(
            train_frame, test_frame, vertices, gt_track_3d, idx, mask
        )
        writer.writerow([case_name, train_track_error, test_track_error])
    file.close()
