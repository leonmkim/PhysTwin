"""Headless-safe helpers for Open3D point-cloud vectors."""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d


def vec3d(arr) -> o3d.utility.Vector3dVector:  # type: ignore[name-defined]
    """Return Vector3dVector from contiguous float64 (N, 3) input."""
    values = np.asarray(arr)
    if values.size == 0:
        return o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
    values = np.ascontiguousarray(values, dtype=np.float64)
    return o3d.utility.Vector3dVector(values)


def radius_neighbor_indices(points: np.ndarray, query: np.ndarray, radius: float) -> list[int]:
    """Radius search without Open3D KDTreeFlann (headless-safe)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(np.ascontiguousarray(points, dtype=np.float64))
    query_pt = np.ascontiguousarray(query, dtype=np.float64)
    return list(tree.query_ball_point(query_pt, radius))


def build_radius_index(points: np.ndarray):
    """Build a reusable scipy cKDTree for repeated radius queries."""
    from scipy.spatial import cKDTree

    return cKDTree(np.ascontiguousarray(points, dtype=np.float64))


def query_radius_neighbors(tree, query: np.ndarray, radius: float) -> list[int]:
    query_pt = np.ascontiguousarray(query, dtype=np.float64)
    return list(tree.query_ball_point(query_pt, radius))


def search_hybrid_neighbors(
    tree,
    query: np.ndarray,
    radius: float,
    max_nn: int,
) -> tuple[int, list[int], list[float]]:
    """Headless-safe replacement for Open3D ``KDTreeFlann.search_hybrid_vector_3d``."""
    query_pt = np.ascontiguousarray(query, dtype=np.float64)
    indices = list(tree.query_ball_point(query_pt, radius))
    if not indices:
        return 0, [], []
    data = np.ascontiguousarray(tree.data, dtype=np.float64)
    idx_arr = np.asarray(indices, dtype=int)
    dists = np.linalg.norm(data[idx_arr] - query_pt, axis=1)
    order = np.argsort(dists)
    cap = min(max_nn, len(order))
    selected = [indices[order[i]] for i in range(cap)]
    dist_sq = (dists[order[:cap]] ** 2).tolist()
    return cap, selected, dist_sq


def vec3i(arr) -> o3d.utility.Vector3iVector:  # type: ignore[name-defined]
    """Return Vector3iVector from contiguous int32 (M, 3) input."""
    values = np.asarray(arr)
    if values.size == 0:
        return o3d.utility.Vector3iVector(np.empty((0, 3), dtype=np.int32))
    values = np.ascontiguousarray(values, dtype=np.int32)
    return o3d.utility.Vector3iVector(values)


def transform_mesh_vertices(mesh: o3d.geometry.TriangleMesh, transform_4x4) -> o3d.geometry.TriangleMesh:
    """Apply SE(3) to mesh vertices without Open3D ``TriangleMesh.transform`` (headless-safe)."""
    transform = np.ascontiguousarray(transform_4x4, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return mesh
    verts_h = np.hstack([verts, np.ones((len(verts), 1), dtype=np.float64)])
    verts_world = (transform @ verts_h.T).T[:, :3]
    mesh.vertices = vec3d(verts_world)
    return mesh


LOCKED_NUMPY_VERSION = (1, 26, 4)
LOCKED_OPENCV_VERSION_PREFIX = (4, 11, 0)
LOCKED_OPEN3D_VERSION = (0, 19, 0)
MIN_OPEN3D_VERSION = (0, 19, 0)
BLOCKED_OPEN3D_VERSIONS = frozenset({(0, 17, 0)})

CODEC_FALLBACK_ORDER = ("avc1", "mp4v")


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(version).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def check_diagnostic_runtime(*, require_xvfb: bool = False) -> dict[str, str]:
    """Validate NumPy/OpenCV/Open3D versions for author-native diagnostic rendering."""
    import cv2

    numpy_version = _parse_version_tuple(np.__version__)
    opencv_version = _parse_version_tuple(cv2.__version__)
    open3d_version = _parse_version_tuple(o3d.__version__)

    if numpy_version[:3] != LOCKED_NUMPY_VERSION:
        raise RuntimeError(
            "Diagnostic rendering requires NumPy "
            f"{'.'.join(map(str, LOCKED_NUMPY_VERSION))}; got {np.__version__}"
        )
    if opencv_version[:3] != LOCKED_OPENCV_VERSION_PREFIX:
        raise RuntimeError(
            "Diagnostic rendering requires OpenCV 4.11.x; "
            f"got {cv2.__version__}"
        )
    if open3d_version in BLOCKED_OPEN3D_VERSIONS or open3d_version < MIN_OPEN3D_VERSION:
        raise RuntimeError(
            "Diagnostic rendering requires Open3D "
            f"{'.'.join(map(str, LOCKED_OPEN3D_VERSION))} or newer; "
            f"got {o3d.__version__} (0.17 is known to segfault under Xvfb)"
        )

    display = os.environ.get("DISPLAY", "")
    if require_xvfb and not display:
        raise RuntimeError(
            "Author-native Open3D rendering requires an X display (use xvfb-run -a)"
        )

    return {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "open3d": o3d.__version__,
        "display": display or "(none)",
    }


def create_checked_visualizer(
    *,
    width: int,
    height: int,
    visible: bool = False,
    window_name: str = "PhysTwinDiagnostic",
) -> o3d.visualization.Visualizer:
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(
        window_name=window_name,
        width=int(width),
        height=int(height),
        visible=bool(visible),
    )
    if not created:
        raise RuntimeError(
            f"Open3D Visualizer.create_window failed ({width}x{height}, visible={visible})"
        )
    return vis


def capture_visualizer_frame(vis: o3d.visualization.Visualizer) -> np.ndarray:
    vis.poll_events()
    vis.update_renderer()
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    return (frame * 255).astype(np.uint8)


def destroy_visualizer(vis: o3d.visualization.Visualizer | None) -> None:
    if vis is None:
        return
    try:
        vis.destroy_window()
    except Exception:
        pass


def create_mp4_writer(
    output_path: str | os.PathLike[str],
    *,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter, str]:
    import cv2

    path = str(output_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    last_error: str | None = None
    for codec in CODEC_FALLBACK_ORDER:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer, codec
        writer.release()
        last_error = codec

    raise RuntimeError(
        f"Failed to open cv2.VideoWriter for {path} with codecs "
        f"{list(CODEC_FALLBACK_ORDER)} (last tried: {last_error})"
    )


def release_mp4_writer(writer: object | None) -> None:
    if writer is None:
        return
    try:
        writer.release()
    except Exception:
        pass


def probe_open3d_calibrated_camera_under_xvfb(
    *,
    width: int = 64,
    height: int = 48,
) -> dict[str, object]:
    """Exercise Open3D window, camera parameters, capture, and MP4 encoding under Xvfb."""
    check_diagnostic_runtime(require_xvfb=True)
    vis = None
    probe_path = os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        "phystwin_o3d_calibrated_probe.mp4",
    )
    try:
        vis = create_checked_visualizer(width=width, height=height, visible=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(width),
            int(height),
            525.0,
            525.0,
            float(width) / 2.0,
            float(height) / 2.0,
        )
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = intrinsic
        parameters.extrinsic = np.eye(4, dtype=np.float64)
        view_control = vis.get_view_control()
        view_control.convert_from_pinhole_camera_parameters(parameters)

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(mesh)
        frame = capture_visualizer_frame(vis)
        if frame.size == 0:
            raise RuntimeError("Open3D calibrated-camera probe captured an empty frame")

        writer, codec = create_mp4_writer(
            probe_path,
            fps=1.0,
            width=width,
            height=height,
        )
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        release_mp4_writer(writer)

        if not os.path.isfile(probe_path) or os.path.getsize(probe_path) <= 0:
            raise RuntimeError("Open3D calibrated-camera probe produced an empty MP4")

        return {
            "pass": True,
            "codec": codec,
            "frame_shape": list(frame.shape),
            "probe_path": probe_path,
        }
    finally:
        destroy_visualizer(vis)


def probe_open3d_visualizer_under_xvfb(*, width: int = 64, height: int = 48) -> dict[str, object]:
    """Lightweight Open3D window + capture probe for diagnostic preflight."""
    return probe_open3d_calibrated_camera_under_xvfb(width=width, height=height)
