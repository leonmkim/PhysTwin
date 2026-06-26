"""Case-level IO backends for PhysTwin data_process pipelines.

Default backend ``phystwin_file_tree`` preserves legacy on-disk layout.
``converted_session`` is explicit opt-in and wraps ``dataset_converter.session_loader``.
"""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

DEFAULT_IO_BACKEND = "phystwin_file_tree"
ZED_DEPTH_STREAM_SUFFIX = "depth_zed_sdk_neural_plus"
WORLD_TRANSFORM_NONE = "none"
WORLD_TRANSFORM_PHYS_TWIN_Z_UP = "phys_twin_z_up"
WORLD_TRANSFORM_CHOICES = (WORLD_TRANSFORM_NONE, WORLD_TRANSFORM_PHYS_TWIN_Z_UP)


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
    start_sync_index: int = 0
    end_sync_index_exclusive: int | None = None
    world_transform: str = WORLD_TRANSFORM_NONE

    def __post_init__(self) -> None:
        if not self.camera_serials:
            raise ValueError("camera_serials must not be empty")
        parse_world_transform(self.world_transform)
        if self.target_fps is not None and self.stride is not None:
            raise ValueError("Specify at most one of target_fps or stride")
        if self.stride is not None and int(self.stride) < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.target_fps is not None and float(self.target_fps) <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if int(self.start_sync_index) < 0:
            raise ValueError(
                f"start_sync_index must be >= 0, got {self.start_sync_index}"
            )
        if self.end_sync_index_exclusive is not None and int(
            self.end_sync_index_exclusive
        ) <= int(self.start_sync_index):
            raise ValueError(
                "end_sync_index_exclusive must be greater than start_sync_index "
                f"({self.start_sync_index} >= {self.end_sync_index_exclusive})"
            )
        non_default_window = int(self.start_sync_index) != 0 or (
            self.end_sync_index_exclusive is not None
        )
        if self.target_fps is not None and non_default_window:
            raise ValueError(
                "target_fps cannot be combined with a native sync-index window "
                "(start_sync_index != 0 or end_sync_index_exclusive is set). "
                "Window indices refer to the native anchor-overlap grid; use stride "
                "instead of target_fps when trimming converted sessions."
            )

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
        start = int(descriptor.start_sync_index)
        end = (
            int(descriptor.end_sync_index_exclusive)
            if descriptor.end_sync_index_exclusive is not None
            else raw_count
        )
        if not (0 <= start < end <= raw_count):
            raise ValueError(
                "sync-index window must satisfy 0 <= start < end <= raw_count "
                f"(start={start}, end={end}, raw_count={raw_count})"
            )
        self._start = start
        window_count = end - start
        effective = (window_count + self._stride - 1) // self._stride
        if descriptor.max_frames is not None:
            effective = min(effective, int(descriptor.max_frames))
        self._frame_count = effective

        wT_path = Path(descriptor.session_path) / "metadata" / "calibration_package" / "world_T_camera_by_serial.npz"
        self._world_T_camera_by_serial = np.load(wT_path)
        self._world_transform_name = parse_world_transform(descriptor.world_transform)
        self._world_T_pt_from_conv = resolve_world_transform_matrix(
            self._world_transform_name
        )
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
        """Exposed frame count after optional sync-index window, stride, and max_frames."""
        return self._frame_count

    def window_start_sync_index(self) -> int:
        return self._start

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
        return self._start + int(frame_idx) * self._stride

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
        c2w = np.asarray(self._world_T_camera_by_serial[serial], dtype=np.float64)
        if self._world_T_pt_from_conv is not None:
            c2w = self._world_T_pt_from_conv @ c2w
        return c2w

    def world_transform_name(self) -> str:
        return self._world_transform_name

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


def parse_world_transform(value: str | None) -> str:
    name = (value or WORLD_TRANSFORM_NONE).strip()
    if name not in WORLD_TRANSFORM_CHOICES:
        raise ValueError(
            f"world_transform must be one of {WORLD_TRANSFORM_CHOICES}, got {name!r}"
        )
    return name


def resolve_world_transform_matrix(name: str) -> np.ndarray | None:
    """Return 4x4 T_pt_from_conv, or None when name is ``none``."""
    parsed = parse_world_transform(name)
    if parsed == WORLD_TRANSFORM_NONE:
        return None
    if parsed == WORLD_TRANSFORM_PHYS_TWIN_Z_UP:
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"unsupported world_transform: {parsed!r}")


def _read_window_yaml(
    path: str | Path,
    *,
    expected_session_id: str | None = None,
) -> tuple[int, int]:
    import yaml

    window_path = Path(path)
    if not window_path.is_file():
        raise FileNotFoundError(f"window yaml not found: {window_path}")
    with open(window_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"window yaml must parse to a mapping: {window_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"window yaml schema_version must be 1, got {payload.get('schema_version')!r}"
        )
    if payload.get("kind") != "phystwin_interaction_window":
        raise ValueError(
            f"window yaml kind must be phystwin_interaction_window, got {payload.get('kind')!r}"
        )
    session = payload.get("session")
    if isinstance(session, dict) and expected_session_id is not None:
        yaml_session_id = session.get("session_id")
        if yaml_session_id is not None and str(yaml_session_id) != str(expected_session_id):
            warnings.warn(
                f"window yaml session_id {yaml_session_id!r} differs from "
                f"converted session directory {expected_session_id!r}",
                stacklevel=2,
            )
    resolved = payload.get("resolved_window")
    if not isinstance(resolved, dict):
        raise ValueError(f"window yaml missing resolved_window mapping: {window_path}")
    if "start_sync_index" not in resolved or "end_sync_index_exclusive" not in resolved:
        raise ValueError(
            "window yaml resolved_window must include start_sync_index and "
            f"end_sync_index_exclusive: {window_path}"
        )
    start = resolved["start_sync_index"]
    end = resolved["end_sync_index_exclusive"]
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError(f"start_sync_index must be int, got {start!r}")
    if not isinstance(end, int) or isinstance(end, bool):
        raise ValueError(f"end_sync_index_exclusive must be int, got {end!r}")
    if start >= end:
        raise ValueError(
            f"window yaml requires start_sync_index < end_sync_index_exclusive "
            f"({start} >= {end})"
        )
    return start, end


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
    window_yaml: str | Path | None = None,
    start_sync_index: int | None = None,
    end_sync_index_exclusive: int | None = None,
    world_transform: str | None = None,
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
        session_path = Path(converted_session_path)
        yaml_start: int | None = None
        yaml_end: int | None = None
        if window_yaml is not None:
            yaml_start, yaml_end = _read_window_yaml(
                window_yaml,
                expected_session_id=session_path.name,
            )
        resolved_start = (
            int(start_sync_index)
            if start_sync_index is not None
            else (yaml_start if yaml_start is not None else 0)
        )
        resolved_end = (
            int(end_sync_index_exclusive)
            if end_sync_index_exclusive is not None
            else yaml_end
        )
        descriptor = ConvertedSessionDescriptor(
            session_path=session_path,
            camera_serials=tuple(serials),
            anchor_serial=anchor_serial,
            anchor_stream_id=anchor_stream_id,
            target_fps=target_fps,
            stride=stride,
            max_frames=max_frames,
            start_sync_index=resolved_start,
            end_sync_index_exclusive=resolved_end,
            world_transform=parse_world_transform(world_transform),
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
    parser.add_argument(
        "--window-yaml",
        type=str,
        default=None,
        help="PhysTwin interaction-window sidecar (metadata/phystwin_window.yaml).",
    )
    parser.add_argument(
        "--start-sync-index",
        type=int,
        default=None,
        help="Native anchor-overlap sync index (inclusive). Overrides YAML start when set.",
    )
    parser.add_argument(
        "--end-sync-index-exclusive",
        type=int,
        default=None,
        help="Native anchor-overlap sync index (exclusive). Overrides YAML end when set.",
    )
    parser.add_argument(
        "--world-transform",
        type=str,
        default=WORLD_TRANSFORM_NONE,
        choices=list(WORLD_TRANSFORM_CHOICES),
        help=(
            "Optional rigid world-frame remap for converted_session c2w only "
            f"(default: {WORLD_TRANSFORM_NONE})."
        ),
    )


def derive_sidecars(
    descriptor: ConvertedSessionDescriptor,
    output_dir: Path,
    *,
    window_yaml: str | Path | None = None,
) -> None:
    """Write minimal PhysTwin case sidecars (metadata, calibrate, color/0/0.png, shape/)."""
    materialize_phys_twin_sidecars(
        descriptor,
        output_dir,
        window_yaml=window_yaml,
    )


def materialize_phys_twin_sidecars(
    descriptor: ConvertedSessionDescriptor,
    output_dir: Path,
    *,
    window_yaml: str | Path | None = None,
) -> None:
    """Scratch sidecars for shape-prior / align without a full RGB cache."""
    backend = ConvertedSessionBackend(descriptor)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_dir = output_dir / "color" / "0"
    color_dir.mkdir(parents=True, exist_ok=True)
    rgb0 = backend.get_rgb(0, 0)
    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(color_dir / "0.png"), bgr0):
        raise OSError(f"Failed to write reference color frame: {color_dir / '0.png'}")

    (output_dir / "shape" / "matching").mkdir(parents=True, exist_ok=True)

    intrinsics = [backend.get_intrinsics(cam_id).tolist() for cam_id in backend.camera_ids()]
    c2ws = [backend.get_c2w(cam_id).tolist() for cam_id in backend.camera_ids()]
    metadata: dict[str, Any] = {
        "intrinsics": intrinsics,
        "WH": [int(rgb0.shape[1]), int(rgb0.shape[0])],
        "frame_num": backend.frame_count(),
        "fps": backend.fps(),
        "serial_numbers": backend.serial_numbers(),
        "source": "converted_session",
        "session_path": str(descriptor.session_path),
        "camera_serials": list(descriptor.camera_serials),
        "world_transform_name": backend.world_transform_name(),
    }
    world_T = resolve_world_transform_matrix(backend.world_transform_name())
    if world_T is not None:
        metadata["world_transform_matrix"] = world_T.tolist()
    if window_yaml is not None:
        window_path = Path(window_yaml)
        metadata["window_yaml"] = str(window_path)
        start, end = _read_window_yaml(
            window_path,
            expected_session_id=descriptor.session_path.name,
        )
        metadata["start_sync_index"] = start
        metadata["end_sync_index_exclusive"] = end
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    with open(output_dir / "calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)


def _jpeg_window_frame_indices(
    backend: CaseIOBackend,
    *,
    max_frames: int | None,
    stride: int,
) -> list[int]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    indices = list(range(0, backend.frame_count(), stride))
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    return indices


def materialize_rgb_window_as_jpegs(
    backend: CaseIOBackend,
    cam_id: int,
    out_dir: str | Path,
    *,
    max_frames: int | None = None,
    stride: int = 1,
) -> int:
    """Write backend RGB frames as zero-padded JPEGs for SAM2 directory input."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    frame_indices = _jpeg_window_frame_indices(
        backend,
        max_frames=max_frames,
        stride=stride,
    )
    for out_idx, frame_idx in enumerate(frame_indices):
        rgb = backend.get_rgb(cam_id, frame_idx)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        filename = f"{out_idx:05d}.jpg"
        if not cv2.imwrite(
            str(out_path / filename),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100],
        ):
            raise OSError(f"Failed to write JPEG: {out_path / filename}")
    written = sorted(
        p.name
        for p in out_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    expected = [f"{i:05d}.jpg" for i in range(len(frame_indices))]
    if written != expected:
        raise RuntimeError(
            f"JPEG lexical order mismatch: expected {expected}, got {written}"
        )
    return len(frame_indices)


@contextmanager
def transient_jpeg_window(
    backend: CaseIOBackend,
    cam_id: int,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    root: str | Path = "/tmp",
    keep: bool = False,
) -> Iterator[tuple[Path, int]]:
    """Materialize a transient JPEG directory; delete on exit unless keep=True."""
    frame_dir = Path(
        tempfile.mkdtemp(prefix="phystwin_jpeg_window_", dir=str(root))
    )
    try:
        frame_count = materialize_rgb_window_as_jpegs(
            backend,
            cam_id,
            frame_dir,
            max_frames=max_frames,
            stride=stride,
        )
        yield frame_dir, frame_count
    finally:
        if not keep and frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)
