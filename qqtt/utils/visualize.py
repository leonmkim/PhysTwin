import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg
from .logger import logger
import pyrender
import trimesh

try:
    from data_process.o3d_utils import (
        capture_visualizer_frame,
        create_checked_visualizer,
        create_mp4_writer,
        destroy_visualizer,
        release_mp4_writer,
    )
except ImportError:  # pragma: no cover - author layout
    from o3d_utils import (
        capture_visualizer_frame,
        create_checked_visualizer,
        create_mp4_writer,
        destroy_visualizer,
        release_mp4_writer,
    )


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    if save_video and cfg.disable_video_logging:
        logger.info(
            "disable_video_logging is set; skipping Open3D video render "
            f"(save_path={save_path})."
        )
        return None

    FPS = cfg.FPS
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]

    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    elif object_colors.shape[1] < object_points.shape[1]:
        object_colors = np.concatenate(
            [
                object_colors,
                np.ones(
                    (
                        object_colors.shape[0],
                        object_points.shape[1] - object_colors.shape[1],
                        3,
                    )
                )
                * 0.3,
            ],
            axis=1,
        )

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")
    if save_video and not save_path:
        raise ValueError("save_path is required when save_video=True")

    vis = None
    video_writer = None
    selected_codec = None
    try:
        vis = create_checked_visualizer(
            width=width,
            height=height,
            visible=bool(visualize),
            window_name="PhysTwinVisualizePC",
        )

        if save_video:
            video_writer, selected_codec = create_mp4_writer(
                save_path,
                fps=FPS,
                width=width,
                height=height,
            )
            logger.info(
                f"Open3D video render using codec {selected_codec!r} -> {save_path}"
            )

        controller_meshes = []
        prev_center = []
        render_object_pcd = None

        for i in range(object_points.shape[0]):
            object_pcd = o3d.geometry.PointCloud()
            if object_visibilities is None:
                object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
                object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
            else:
                visible_idx = np.where(object_visibilities[i])[0]
                object_pcd.points = o3d.utility.Vector3dVector(
                    object_points[i, visible_idx, :]
                )
                object_pcd.colors = o3d.utility.Vector3dVector(
                    object_colors[i, visible_idx, :]
                )

            if i == 0:
                render_object_pcd = object_pcd
                vis.add_geometry(render_object_pcd)
                if controller_points is not None:
                    for j in range(controller_points.shape[1]):
                        origin = controller_points[i, j]
                        controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                            radius=0.01
                        ).translate(origin)
                        controller_mesh.compute_vertex_normals()
                        controller_mesh.paint_uniform_color([1, 0, 0])
                        controller_meshes.append(controller_mesh)
                        vis.add_geometry(controller_meshes[-1])
                        prev_center.append(origin)
                view_control = vis.get_view_control()
                camera_params = o3d.camera.PinholeCameraParameters()
                intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                    width, height, intrinsic
                )
                camera_params.intrinsic = intrinsic_parameter
                camera_params.extrinsic = w2c
                view_control.convert_from_pinhole_camera_parameters(
                    camera_params, allow_arbitrary=True
                )
            else:
                render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
                render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
                vis.update_geometry(render_object_pcd)
                if controller_points is not None:
                    for j in range(controller_points.shape[1]):
                        origin = controller_points[i, j]
                        controller_meshes[j].translate(origin - prev_center[j])
                        vis.update_geometry(controller_meshes[j])
                        prev_center[j] = origin

            if save_video and video_writer is not None:
                frame = capture_visualizer_frame(vis)
                if cfg.overlay_path is not None:
                    mask = np.all(frame == [255, 255, 255], axis=-1)
                    image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                    overlay = cv2.imread(image_path)
                    if overlay is not None:
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        frame[mask] = overlay[mask]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            if visualize:
                time.sleep(1 / FPS)
    finally:
        release_mp4_writer(video_writer)
        destroy_visualizer(vis)

    return None
