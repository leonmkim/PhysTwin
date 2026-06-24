"""Case-level IO backends for PhysTwin data_process pipelines.

Default backend ``phystwin_file_tree`` preserves legacy on-disk layout.
``converted_session`` is explicit opt-in and wraps ``dataset_converter.session_loader``.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_IO_BACKEND = "phystwin_file_tree"
ZED_DEPTH_STREAM_SUFFIX = "depth_zed_sdk_neural_plus"


@dataclass(frozen=True)
class ConvertedSessionDescriptor:
    """Minimal explicit descriptor for converted-session IO."""

    session_path: Path
    camera_serials: tuple[str, ...]
    anchor_serial: str | None = None
    anchor_stream_id: str | None = None
    depth_stream_suffix: str = ZED_DEPTH_STREAM_SUFFIX
    target_fps: float | None = None
    stride: int | None = None
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.camera_serials:
            raise ValueError("camera_serials must not be empty")
        if self.target_fps is not None and self.stride is not None:
            raise ValueError("Specify at most one of target_fps or stride")
        if self.stride is not None and int(self.stride) < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.target_fps is not None and float(self.target_fps) <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")

    @property
    def anchor_color_stream_id(self) -> str:
        if self.anchor_stream_id is not None:
            return self.anchor_stream_id
        anchor_serial = self.anchor_serial or self.camera_serials[0]
        return f"zed_{anchor_serial}_left_rgb"

    def color_stream_id(self, cam_id: int) -> str:
        serial = self.camera_serials[cam_id]
        return f"zed_{serial}_left_rgb"

    def depth_stream_id(self, cam_id: int) -> str:
        serial = self.camera_serials[cam_id]
        return f"zed_{serial}_{self.depth_stream_suffix}"

    def serial_mapping(self) -> dict[int, str]:
        return {i: serial for i, serial in enumerate(self.camera_serials)}


class CaseIOBackend(ABC):
    """Common per-case interface for RGB-D frame access."""

    @abstractmethod
    def backend_name(self) -> str:
        ...

    @abstractmethod
    def camera_ids(self) -> list[int]:
        ...

    @abstractmethod
    def frame_count(self) -> int:
        ...

    @abstractmethod
    def fps(self) -> float:
        ...

    @abstractmethod
    def timestamp_ns(self, frame_idx: int) -> int:
        ...

    @abstractmethod
    def get_rgb(self, cam_id: int, frame_idx: int) -> np.ndarray:
        """Return HxWx3 uint8 RGB."""

    @abstractmethod
    def get_depth_mm(self, cam_id: int, frame_idx: int) -> np.ndarray:
        """Return HxW depth in millimetres (uint16 or float)."""

    @abstractmethod
    def get_intrinsics(self, cam_id: int) -> np.ndarray:
        """Return 3x3 pinhole intrinsics."""

    @abstractmethod
    def get_c2w(self, cam_id: int) -> np.ndarray:
        """Return 4x4 camera-to-world transform."""

    def source_frame_index(self, cam_id: int, frame_idx: int) -> int | None:
        return None

    def valid(self, cam_id: int, frame_idx: int) -> bool | None:
        return None

    def serial_numbers(self) -> list[str]:
        return [str(i) for i in self.camera_ids()]


class PhystwinFileTreeBackend(CaseIOBackend):
    """Legacy PhysTwin case directory: metadata.json + calibrate.pkl + color/depth trees."""

    def __init__(
        self,
        base_path: str | Path,
        case_name: str,
        *,
        max_frames: int | None = None,
    ) -> None:
        self._case_path = Path(base_path) / case_name
        if not self._case_path.is_dir():
            raise FileNotFoundError(f"Case directory not found: {self._case_path}")
        with open(self._case_path / "metadata.json", "r", encoding="utf-8") as f:
            self._metadata: dict[str, Any] = json.load(f)
        self._intrinsics = np.asarray(self._metadata["intrinsics"], dtype=np.float64)
        self._c2ws = pickle.load(open(self._case_path / "calibrate.pkl", "rb"))
        self._num_cam = int(len(self._intrinsics))
        self._frame_num = int(self._metadata["frame_num"])
        if max_frames is not None:
            self._frame_num = min(self._frame_num, int(max_frames))
        wh = self._metadata.get("WH")
        self._wh = tuple(wh) if wh is not None else None

    def backend_name(self) -> str:
        return DEFAULT_IO_BACKEND

    def camera_ids(self) -> list[int]:
        return list(range(self._num_cam))

    def frame_count(self) -> int:
        return self._frame_num

    def fps(self) -> float:
        if "fps" not in self._metadata:
            raise KeyError(
                f"metadata.json in {self._case_path} is missing required key 'fps'"
            )
        return float(self._metadata["fps"])

    def timestamp_ns(self, frame_idx: int) -> int:
        fps = self.fps()
        return int(round(frame_idx * (1_000_000_000.0 / fps)))

    def get_rgb(self, cam_id: int, frame_idx: int) -> np.ndarray:
        path = self._case_path / "color" / str(cam_id) / f"{frame_idx}.png"
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(f"Missing color frame: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_depth_mm(self, cam_id: int, frame_idx: int) -> np.ndarray:
        path = self._case_path / "depth" / str(cam_id) / f"{frame_idx}.npy"
        depth = np.load(path)
        return np.asarray(depth)

    def get_intrinsics(self, cam_id: int) -> np.ndarray:
        return np.asarray(self._intrinsics[cam_id], dtype=np.float64)

    def get_c2w(self, cam_id: int) -> np.ndarray:
        return np.asarray(self._c2ws[cam_id], dtype=np.float64)

    def serial_numbers(self) -> list[str]:
        serials = self._metadata.get("serial_numbers")
        if serials is not None:
            return [str(s) for s in serials]
        return super().serial_numbers()


class ConvertedSessionBackend(CaseIOBackend):
    """Converted belt_perception session via ``dataset_converter.session_loader``."""

    def __init__(self, descriptor: ConvertedSessionDescriptor) -> None:
        from dataset_converter.session_loader import (
            AnchorStreamSyncPolicy,
            ConvertedSessionSynchronizedLoader,
            FixedRateSyncPolicy,
            SessionIndex,
            TimeRange,
        )

        self._descriptor = descriptor
        self._serial_by_cam = descriptor.serial_mapping()
        self._color_streams = [descriptor.color_stream_id(i) for i in self._serial_by_cam]
        self._depth_streams = [descriptor.depth_stream_id(i) for i in self._serial_by_cam]
        self._streams = [x for pair in zip(self._color_streams, self._depth_streams) for x in pair]
        self._anchor_stream = descriptor.anchor_color_stream_id

        self._index = SessionIndex.build(descriptor.session_path)
        if descriptor.target_fps is not None:
            sync_policy = FixedRateSyncPolicy(
                fps=float(descriptor.target_fps),
                time_range=TimeRange.overlap(),
            )
        else:
            sync_policy = AnchorStreamSyncPolicy(
                anchor_stream_id=self._anchor_stream,
                time_range=TimeRange.overlap(),
            )

        self._loader = ConvertedSessionSynchronizedLoader(
            descriptor.session_path,
            streams=self._streams,
            sync_policy=sync_policy,
            dimension_order="BTVHWC",
            dtype_policy={"color": "uint8", "depth": "uint16_mm"},
            chunk_size=1,
        )
        self._sync_policy_name = type(sync_policy).__name__
        self._stride = int(descriptor.stride) if descriptor.stride is not None else 1
        raw_count = int(self._loader.num_sync_samples())
        self._raw_frame_count = raw_count
        effective = (raw_count + self._stride - 1) // self._stride
        if descriptor.max_frames is not None:
            effective = min(effective, int(descriptor.max_frames))
        self._frame_count = effective

        wT_path = Path(descriptor.session_path) / "metadata" / "calibration_package" / "world_T_camera_by_serial.npz"
        self._world_T_camera_by_serial = np.load(wT_path)
        self._fps_value = self._estimate_native_fps()
        self._sample_cache: dict[int, Any] = {}

    def backend_name(self) -> str:
        return "converted_session"

    def sync_policy_name(self) -> str:
        return self._sync_policy_name

    def anchor_stream_id(self) -> str:
        return self._anchor_stream

    def camera_ids(self) -> list[int]:
        return list(self._serial_by_cam.keys())

    def frame_count(self) -> int:
        """Anchor-frame samples in the all-camera overlap window.

        Counts anchor-stream sync samples within ``TimeRange.overlap()``, then
        applies optional ``stride`` and ``max_frames`` caps from the descriptor.
        Full-anchor range or ``target_fps`` resampling are separate explicit modes.
        """
        return self._frame_count

    def raw_sync_sample_count(self) -> int:
        return self._raw_frame_count

    def fps(self) -> float:
        return self._fps_value

    def serial_numbers(self) -> list[str]:
        return list(self._descriptor.camera_serials)

    def serial_mapping(self) -> dict[int, str]:
        return dict(self._serial_by_cam)

    def _estimate_native_fps(self) -> float:
        ts = self._index.timestamps_by_stream[self._anchor_stream].astype(np.int64)
        if len(ts) < 2:
            return 0.0
        median_dt_ns = float(np.median(np.diff(ts)))
        return 1e9 / median_dt_ns

    def _loader_index(self, frame_idx: int) -> int:
        if frame_idx < 0 or frame_idx >= self._frame_count:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {self._frame_count})")
        return int(frame_idx) * self._stride

    def _get_sample(self, frame_idx: int):
        loader_idx = self._loader_index(frame_idx)
        if loader_idx not in self._sample_cache:
            self._sample_cache[loader_idx] = self._loader.get_item(loader_idx)
        return self._sample_cache[loader_idx]

    def timestamp_ns(self, frame_idx: int) -> int:
        sample = self._get_sample(frame_idx)
        qts = sample.sync.query_timestamps_ns
        if qts.ndim == 2:
            return int(qts[0, 0])
        return int(qts[0, 0, 0])

    def _array_for_stream(self, sample, stream_id: str) -> np.ndarray:
        arr = np.asarray(sample.data_by_stream[stream_id])
        # BTVHWC with B=T=V=1 -> squeeze to HWC or HW
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def get_rgb(self, cam_id: int, frame_idx: int) -> np.ndarray:
        sample = self._get_sample(frame_idx)
        stream_id = self._color_streams[cam_id]
        rgb = self._array_for_stream(sample, stream_id)
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8, copy=False)
        return rgb

    def get_depth_mm(self, cam_id: int, frame_idx: int) -> np.ndarray:
        sample = self._get_sample(frame_idx)
        stream_id = self._depth_streams[cam_id]
        depth = self._array_for_stream(sample, stream_id)
        return np.asarray(depth)

    def get_intrinsics(self, cam_id: int) -> np.ndarray:
        # Intrinsics are static; use frame 0 sample.
        sample = self._get_sample(0)
        stream_id = self._color_streams[cam_id]
        intr = sample.intrinsics_by_stream[stream_id]
        return np.array(
            [
                [intr.fx, 0.0, intr.cx],
                [0.0, intr.fy, intr.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_c2w(self, cam_id: int) -> np.ndarray:
        serial = self._serial_by_cam[cam_id]
        return np.asarray(self._world_T_camera_by_serial[serial], dtype=np.float64)

    def source_frame_index(self, cam_id: int, frame_idx: int) -> int | None:
        sample = self._get_sample(frame_idx)
        stream_id = self._color_streams[cam_id]
        idx = int(sample.sync.source_frame_indices[stream_id][0, 0, 0])
        return idx if idx >= 0 else None

    def valid(self, cam_id: int, frame_idx: int) -> bool | None:
        sample = self._get_sample(frame_idx)
        stream_id = self._color_streams[cam_id]
        return bool(sample.sync.valid[stream_id][0, 0, 0])


def parse_camera_serials(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def create_io_backend(
    *,
    io_backend: str = DEFAULT_IO_BACKEND,
    base_path: str | Path | None = None,
    case_name: str | None = None,
    converted_session_path: str | Path | None = None,
    camera_serials: str | list[str] | None = None,
    anchor_serial: str | None = None,
    anchor_stream_id: str | None = None,
    target_fps: float | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
) -> CaseIOBackend:
    backend = (io_backend or DEFAULT_IO_BACKEND).strip()
    if backend == DEFAULT_IO_BACKEND:
        if base_path is None or case_name is None:
            raise ValueError("base_path and case_name are required for phystwin_file_tree backend")
        return PhystwinFileTreeBackend(base_path, case_name, max_frames=max_frames)
    if backend == "converted_session":
        if converted_session_path is None:
            raise ValueError("--converted-session-path is required for converted_session backend")
        serials = (
            list(camera_serials)
            if isinstance(camera_serials, list)
            else parse_camera_serials(camera_serials)
        )
        if not serials:
            raise ValueError("--camera-serials is required for converted_session backend")
        descriptor = ConvertedSessionDescriptor(
            session_path=Path(converted_session_path),
            camera_serials=tuple(serials),
            anchor_serial=anchor_serial,
            anchor_stream_id=anchor_stream_id,
            target_fps=target_fps,
            stride=stride,
            max_frames=max_frames,
        )
        return ConvertedSessionBackend(descriptor)
    raise ValueError(f"Unknown io_backend: {backend!r}")


def add_io_backend_args(parser) -> None:
    parser.add_argument(
        "--io-backend",
        type=str,
        default=DEFAULT_IO_BACKEND,
        choices=[DEFAULT_IO_BACKEND, "converted_session"],
        help="Case IO backend (default: phystwin_file_tree).",
    )
    parser.add_argument(
        "--converted-session-path",
        type=str,
        default=None,
        help="Converted session root (required for converted_session backend).",
    )
    parser.add_argument(
        "--camera-serials",
        type=str,
        default=None,
        help="Comma-separated ZED serials mapped to camera ids 0..N-1.",
    )
    parser.add_argument(
        "--anchor-serial",
        type=str,
        default=None,
        help="Anchor camera serial for native sync (default: first serial).",
    )
    parser.add_argument(
        "--anchor-stream-id",
        type=str,
        default=None,
        help="Explicit anchor color stream id (overrides --anchor-serial).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Opt-in downsampling via FixedRateSyncPolicy (default: native anchor cadence).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Opt-in integer stride over native sync samples (default: none).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames processed (smoke tests).",
    )


def derive_sidecars(
    descriptor: ConvertedSessionDescriptor,
    output_dir: Path,
) -> None:
    """Optional scratch sidecars (metadata.json + calibrate.pkl). Not used by default."""
    backend = ConvertedSessionBackend(descriptor)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = [backend.get_intrinsics(cam_id).tolist() for cam_id in backend.camera_ids()]
    c2ws = [backend.get_c2w(cam_id).tolist() for cam_id in backend.camera_ids()]
    rgb0 = backend.get_rgb(0, 0)
    metadata = {
        "intrinsics": intrinsics,
        "WH": [int(rgb0.shape[1]), int(rgb0.shape[0])],
        "frame_num": backend.frame_count(),
        "fps": backend.fps(),
        "serial_numbers": backend.serial_numbers(),
        "source": "converted_session",
        "session_path": str(descriptor.session_path),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    with open(output_dir / "calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)
