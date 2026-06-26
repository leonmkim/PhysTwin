# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)

import glob
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from utils.visualizer import Visualizer

from data_process.io_backend import add_io_backend_args, create_io_backend

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
add_io_backend_args(parser)
parser.add_argument(
    "--camera-idx",
    type=int,
    default=None,
    help="Camera index for converted_session backend (required in converted mode).",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
num_cam = 3
device = "cuda"


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def track_one_camera(cam_id: int, video: torch.Tensor, case_dir: str | Path, device: str) -> None:
    """Run CoTracker for one camera and write cotracker/{cam_id}.npz."""
    case_dir = Path(case_dir)
    exist_dir(case_dir / "cotracker")

    mask_paths = glob.glob(str(case_dir / "mask" / str(cam_id) / "*" / "0.png"))
    mask = None
    for mask_path in mask_paths:
        current_mask = read_mask(mask_path)
        if mask is None:
            mask = current_mask
        else:
            mask = np.logical_or(mask, current_mask)
    if mask is None or not np.any(mask):
        raise RuntimeError(
            f"No nonzero union mask for camera {cam_id} under {case_dir / 'mask' / str(cam_id)}"
        )

    query_pixels = np.argwhere(mask)
    # Revert x and y
    query_pixels = query_pixels[:, ::-1]
    query_pixels = np.concatenate(
        [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
    )
    query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
    # Randomly select 5000 query points
    query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

    cotracker = torch.hub.load(
        "facebookresearch/co-tracker", "cotracker3_online"
    ).to(device)
    cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

    for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
        pred_tracks, pred_visibility = cotracker(
            video_chunk=video[:, ind : ind + cotracker.step * 2]
        )  # B T N 2,  B T N 1

    try:
        vis = Visualizer(
            save_dir=str(case_dir / "cotracker"), pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{cam_id}")
    except Exception as exc:
        print(
            f"Warning: cotracker visualization failed for camera {cam_id} "
            f"(npz will still be written): {exc}"
        )

    track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
    visibility_to_save = pred_visibility[0].cpu().numpy()
    np.savez(
        case_dir / "cotracker" / f"{cam_id}.npz",
        tracks=track_to_save,
        visibility=visibility_to_save,
    )


def _legacy_track_all_cameras(case_dir: str | Path) -> None:
    assert len(glob.glob(f"{case_dir}/depth/*")) == num_cam
    exist_dir(f"{case_dir}/cotracker")
    for i in range(num_cam):
        print(f"Processing {i}th camera")
        frames = iio.imread(f"{case_dir}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        track_one_camera(i, video, case_dir, device)


def _converted_session_track(case_dir: str | Path) -> None:
    if args.camera_idx is None:
        raise ValueError(
            "--camera-idx is required when --io-backend converted_session"
        )
    backend = create_io_backend(
        io_backend=args.io_backend,
        base_path=base_path,
        case_name=case_name,
        converted_session_path=args.converted_session_path,
        camera_serials=args.camera_serials,
        anchor_serial=args.anchor_serial,
        anchor_stream_id=args.anchor_stream_id,
        window_yaml=args.window_yaml,
        start_sync_index=args.start_sync_index,
        end_sync_index_exclusive=args.end_sync_index_exclusive,
        target_fps=args.target_fps,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    cam_id = int(args.camera_idx)
    frame_count = backend.frame_count()
    print(
        f"Converted-session dense track: camera_idx={cam_id}, "
        f"frames={frame_count}, window_start={backend.window_start_sync_index()}"
    )
    frames = np.stack(
        [backend.get_rgb(cam_id, t) for t in range(frame_count)],
        axis=0,
    )
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    track_one_camera(cam_id, video, case_dir, device)


if __name__ == "__main__":
    case_dir = f"{base_path}/{case_name}"
    if args.io_backend == "converted_session":
        _converted_session_track(case_dir)
    else:
        _legacy_track_all_cameras(case_dir)
