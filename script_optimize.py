import glob
import json
import subprocess
import sys
from argparse import ArgumentParser

from qqtt.utils.output_dirs import add_experiments_optimization_dir_arg


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    add_experiments_optimization_dir_arg(parser)
    args = parser.parse_args()

    base_path = args.base_path
    dir_names = glob.glob(f"{base_path}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]

        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)

        train_frame = split["train"][1]

        cmd = [
            sys.executable,
            "optimize_cma.py",
            "--base_path",
            base_path,
            "--case_name",
            case_name,
            "--train_frame",
            str(train_frame),
            "--experiments-optimization-dir",
            args.experiments_optimization_dir,
        ]
        subprocess.run(cmd, check=True)
