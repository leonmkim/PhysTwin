"""Headless-safe helpers for Open3D point-cloud vectors."""

from __future__ import annotations

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
