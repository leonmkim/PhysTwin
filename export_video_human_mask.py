import glob
import os
import subprocess
import sys
from argparse import ArgumentParser

TEXT_PROMPT = "human"
CAMERA_NUM = 3


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Export per-camera human masks for visualization and render evaluation. "
            "Writes under --output-path and removes tmp_data only under --base-path."
        ),
    )
    parser.add_argument(
        "--base-path",
        dest="base_path",
        type=str,
        default="./data/different_types",
        help=(
            "Processed case root (default: ./data/different_types). "
            "Use a scratch root such as ./temp_processed_data_uv for validation; "
            "do not point at author symlinked data/ when running exports."
        ),
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        type=str,
        default="./data/different_types_human_mask",
        help="Human mask output root (default: ./data/different_types_human_mask)",
    )
    parser.add_argument(
        "--case-name",
        dest="case_name",
        type=str,
        default=None,
        help="Process only this case (default: all cases under --base-path)",
    )
    return parser.parse_args()


def iter_case_names(base_path, case_name_filter):
    if case_name_filter is not None:
        case_dir = os.path.join(base_path, case_name_filter)
        if os.path.isdir(case_dir):
            yield case_name_filter
        return

    for dir_name in sorted(glob.glob(os.path.join(base_path, "*"))):
        if os.path.isdir(dir_name):
            yield os.path.basename(dir_name)


def remove_tmp_data(base_path, case_name):
    tmp_data = os.path.join(base_path, case_name, "tmp_data")
    if os.path.isdir(tmp_data):
        subprocess.run(["rm", "-rf", tmp_data], check=True)


def main():
    args = parse_args()
    base_path = args.base_path
    output_path = args.output_path

    existDir(output_path)

    for case_name in iter_case_names(base_path, args.case_name):
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        existDir(f"{output_path}/{case_name}")

        depth_glob = glob.glob(f"{base_path}/{case_name}/depth/*")
        assert len(depth_glob) == CAMERA_NUM, (
            f"Expected {CAMERA_NUM} depth entries for {case_name}, found {len(depth_glob)}"
        )

        for camera_idx in range(CAMERA_NUM):
            print(f"Processing {case_name} camera {camera_idx}")
            subprocess.run(
                [
                    sys.executable,
                    "./data_process/segment_util_video.py",
                    "--output_path",
                    f"{output_path}/{case_name}",
                    "--base_path",
                    base_path,
                    "--case_name",
                    case_name,
                    "--TEXT_PROMPT",
                    TEXT_PROMPT,
                    "--camera_idx",
                    str(camera_idx),
                ],
                check=True,
            )
            remove_tmp_data(base_path, case_name)


if __name__ == "__main__":
    main()
