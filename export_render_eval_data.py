import csv
import json
import os
import subprocess
from argparse import ArgumentParser

CONTROLLER_NAME = "hand"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    parser = ArgumentParser(
        description="Package RGB, object masks, and split.json for render evaluation",
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
        default="./data/render_eval_data",
        help="Render eval package output root (default: ./data/render_eval_data)",
    )
    parser.add_argument(
        "--case-name",
        dest="case_name",
        type=str,
        default=None,
        help="Process only this case (default: all cases in data_config.csv)",
    )
    return parser.parse_args()


def export_case(base_path, output_path, case_name):
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    existDir(f"{output_path}/{case_name}")
    existDir(f"{output_path}/{case_name}/mask")
    for i in range(3):
        subprocess.run(
            ["cp", "-r", f"{base_path}/{case_name}/color", f"{output_path}/{case_name}/"],
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
        existDir(f"{output_path}/{case_name}/mask/{i}")
        os.system(
            f"cp -r {base_path}/{case_name}/mask/{i}/{obj_idx}/* "
            f"{output_path}/{case_name}/mask/{i}/"
        )

    subprocess.run(
        ["cp", f"{base_path}/{case_name}/split.json", f"{output_path}/{case_name}/"],
        check=True,
    )


def main():
    args = parse_args()
    base_path = args.base_path
    output_path = args.output_path

    existDir(output_path)

    with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            case_name = row[0]

            if args.case_name is not None and case_name != args.case_name:
                continue

            if not os.path.exists(f"{base_path}/{case_name}"):
                continue

            export_case(base_path, output_path, case_name)


if __name__ == "__main__":
    main()
