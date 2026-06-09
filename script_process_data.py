import csv
import os
import subprocess
import sys
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Batch-run process_data.py for cases listed in data_config.csv",
)
parser.add_argument(
    "--base-path",
    dest="base_path",
    type=str,
    default="./data/different_types",
    help="Root directory containing per-case folders (default: ./data/different_types)",
)
parser.add_argument(
    "--timer-log",
    dest="timer_log",
    type=str,
    default="timer.log",
    help="Timer log path removed at start and written by child process_data.py (default: timer.log)",
)
args = parser.parse_args()

base_path = args.base_path
timer_log = args.timer_log

if os.path.isfile(timer_log):
    os.remove(timer_log)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        cmd = [
            sys.executable,
            "process_data.py",
            "--base_path",
            base_path,
            "--case_name",
            case_name,
            "--category",
            category,
            "--timer-log",
            timer_log,
        ]
        if shape_prior.lower() == "true":
            cmd.append("--shape_prior")

        subprocess.run(cmd, check=True)
