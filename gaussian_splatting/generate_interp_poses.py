import argparse
import numpy as np
import os
import pickle
import scipy.interpolate


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
    """Creates a smooth spline path between input keyframe camera poses.
    Adapted from https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


def generate_closed_loop_interp_poses(
    c2ws: list[np.ndarray] | np.ndarray,
    *,
    n_interp: int = 50,
) -> list[np.ndarray]:
    """Interpolate a closed camera path over all input poses.

    Generates segments 0->1, 1->2, ..., (N-1)->0 and concatenates them.
    """
    poses = np.asarray(c2ws, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected c2ws with shape (N, 4, 4); got {poses.shape}"
        )
    camera_count = int(poses.shape[0])
    if camera_count < 2:
        raise ValueError(
            f"At least 2 camera poses are required; got {camera_count}"
        )

    segments: list[np.ndarray] = []
    for start_idx in range(camera_count):
        end_idx = (start_idx + 1) % camera_count
        pair = np.stack([poses[start_idx], poses[end_idx]], axis=0)[:, :3, :]
        segments.append(generate_interpolated_path(pair, n_interp))
    interp_poses = np.concatenate(segments, axis=0)
    return [
        np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0])])
        for pose in interp_poses
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaussian-data-dir",
        default="./data/gaussian_data",
        help="Read camera_meta.pkl from this root (default: ./data/gaussian_data)",
    )
    parser.add_argument(
        "--interp-poses-output-dir",
        default=None,
        help=(
            "Write interp_poses.pkl under this root per scene. "
            "Default: write into each scene dir under --gaussian-data-dir (upstream)."
        ),
    )
    args = parser.parse_args()

    root_dir = args.gaussian_data_dir
    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        print(f'Processing {scene_name}')
        camera_path = os.path.join(scene_dir, 'camera_meta.pkl')
        with open(camera_path, 'rb') as f:
            camera_meta = pickle.load(f)
        c2ws = camera_meta['c2ws']
        output_poses = generate_closed_loop_interp_poses(c2ws, n_interp=50)
        if args.interp_poses_output_dir:
            out_scene_dir = os.path.join(args.interp_poses_output_dir, scene_name)
            os.makedirs(out_scene_dir, exist_ok=True)
            out_path = os.path.join(out_scene_dir, 'interp_poses.pkl')
        else:
            out_path = os.path.join(scene_dir, 'interp_poses.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(output_poses, f)
