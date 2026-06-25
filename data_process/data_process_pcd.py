# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
from argparse import ArgumentParser

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
    "--no-vis",
    action="store_true",
    help="Skip Open3D visualization (smoke / headless runs).",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

io = create_io_backend(
    io_backend=args.io_backend,
    base_path=base_path,
    case_name=case_name,
    converted_session_path=args.converted_session_path,
    camera_serials=args.camera_serials,
    anchor_serial=args.anchor_serial,
    anchor_stream_id=args.anchor_stream_id,
    target_fps=args.target_fps,
    stride=args.stride,
    max_frames=args.max_frames,
    window_yaml=args.window_yaml,
    start_sync_index=args.start_sync_index,
    end_sync_index_exclusive=args.end_sync_index_exclusive,
)


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


def getPcdFromDepth(depth, intrinsic):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic).T
    points = points.reshape(H, W, 3)
    return points


def get_pcd_from_backend(backend, frame_idx):
    total_points = []
    total_colors = []
    total_masks = []
    for cam_id in backend.camera_ids():
        color = backend.get_rgb(cam_id, frame_idx)
        color = color.astype(np.float32) / 255.0
        depth_mm = backend.get_depth_mm(cam_id, frame_idx)
        depth = np.asarray(depth_mm, dtype=np.float32) / 1000.0
        intrinsic = backend.get_intrinsics(cam_id)
        c2w = backend.get_c2w(cam_id)

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsic,
        )
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 1.5)
        points_flat = points.reshape(-1, 3)
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2w, homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)
    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    print(io.serial_numbers())

    num_cam = len(io.camera_ids())
    frame_num = io.frame_count()

    exist_dir(f"{base_path}/{case_name}/pcd")

    vis = None
    pcd = None
    if not args.no_vis:
        cameras = []
        for cam_id in io.camera_ids():
            intr = io.get_intrinsics(cam_id)
            c2w = io.get_c2w(cam_id)
            camera = getCamera(
                c2w,
                intr[0, 0],
                intr[1, 1],
                intr[0, 2],
                intr[1, 2],
                z_flip=True,
                scale=0.2,
            )
            cameras += camera

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for camera in cameras:
            vis.add_geometry(camera)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coordinate)

    for i in tqdm(range(frame_num)):
        points, colors, masks = get_pcd_from_backend(io, i)

        if vis is not None:
            if i == 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.add_geometry(pcd)
                view_control = vis.get_view_control()
                view_control.set_front([1, 0, -2])
                view_control.set_up([0, 0, -1])
                view_control.set_zoom(1)
            else:
                pcd.points = o3d.utility.Vector3dVector(
                    points.reshape(-1, 3)[masks.reshape(-1)]
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    colors.reshape(-1, 3)[masks.reshape(-1)]
                )
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )

    if vis is not None:
        vis.destroy_window()
