import csv
import json
import os
import pickle
import subprocess
import sys
from argparse import ArgumentParser

CONTROLLER_NAME = "hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    parser = ArgumentParser(
        description="Export first-frame Gaussian training inputs from processed cases",
    )
    parser.add_argument(
        "--base-path",
        dest="base_path",
        type=str,
        default="./data/different_types",
        help="Processed case root (default: ./data/different_types)",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        default="./data/gaussian_data",
        help="Gaussian data output root (default: ./data/gaussian_data)",
    )
    parser.add_argument(
        "--case-name",
        dest="case_name",
        type=str,
        default=None,
        help="Process only this case (default: all cases in data_config.csv)",
    )
    return parser.parse_args()


def export_case(base_path, output_path, case_name, category, shape_prior):
    import numpy as np
    import open3d as o3d

    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    existDir(f"{output_path}/{case_name}")
    for i in range(3):
        subprocess.run(
            [
                "cp",
                f"{base_path}/{case_name}/color/{i}/0.png",
                f"{output_path}/{case_name}/{i}.png",
            ],
            check=True,
        )
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        obj_idx = None
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if obj_idx is not None:
                    raise ValueError("More than one object detected.")
                obj_idx = int(key)
        mask_path = f"{base_path}/{case_name}/mask/{i}/{obj_idx}/0.png"
        subprocess.run(
            ["cp", mask_path, f"{output_path}/{case_name}/mask_{i}.png"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "./data_process/image_upscale.py",
                "--img_path",
                f"{base_path}/{case_name}/color/{i}/0.png",
                "--output_path",
                f"{output_path}/{case_name}/{i}_high.png",
                "--category",
                category,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "./data_process/segment_util_image.py",
                "--img_path",
                f"{output_path}/{case_name}/{i}_high.png",
                "--TEXT_PROMPT",
                category,
                "--output_path",
                f"{output_path}/{case_name}/mask_{i}_high.png",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cp",
                f"{base_path}/{case_name}/depth/{i}/0.npy",
                f"{output_path}/{case_name}/{i}_depth.npy",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "./data_process/segment_util_image.py",
                "--img_path",
                f"{output_path}/{case_name}/{i}.png",
                "--TEXT_PROMPT",
                "human",
                "--output_path",
                f"{output_path}/{case_name}/mask_human_{i}.png",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "./data_process/segment_util_image.py",
                "--img_path",
                f"{output_path}/{case_name}/{i}_high.png",
                "--TEXT_PROMPT",
                "human",
                "--output_path",
                f"{output_path}/{case_name}/mask_human_{i}_high.png",
            ],
            check=True,
        )

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        intrinsics = json.load(f)["intrinsics"]
    camera_meta = {"c2ws": c2ws, "intrinsics": intrinsics}
    with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
        pickle.dump(camera_meta, f)

    if shape_prior.lower() == "true":
        subprocess.run(
            [
                "cp",
                f"{base_path}/{case_name}/shape/matching/final_mesh.glb",
                f"{output_path}/{case_name}/shape_prior.glb",
            ],
            check=True,
        )

    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    processed_mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    pcd_data = np.load(pcd_path)
    with open(processed_mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(3):
        points = pcd_data["points"][i]
        colors = pcd_data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obs_points)
    pcd.colors = o3d.utility.Vector3dVector(obs_colors)
    o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)


def main():
    args = parse_args()
    base_path = args.base_path
    output_path = args.output_path

    existDir(output_path)

    with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            case_name = row[0]
            category = row[1]
            shape_prior = row[2]

            if args.case_name is not None and case_name != args.case_name:
                continue

            if not os.path.exists(f"{base_path}/{case_name}"):
                continue

            export_case(base_path, output_path, case_name, category, shape_prior)


if __name__ == "__main__":
    main()
