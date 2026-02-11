"""
Cost functions for SC-NOD Adaptive 3D Box Search.

Paper equations:
  J(theta) = J3D(theta, Pobj, e) + J2D(theta, bimg)           (Eq. 1)
  J3D = lambda1*Jdensity + lambda2*Jl-shape + lambda3*Jsurface (Eq. 2)
  Jdensity = -Ginside / |Pobj|                                 (Eq. 3-4)
  Jl-shape = point-to-edge distance for 2 nearest top edges    (Eq. 5)
  Jsurface = -min(||(tx,ty)-(ex,ey)||, Csurface)               (Eq. 6)
  J2D = gamma * (1 - IoU2D(proj(theta), bimg))                 (Eq. 7)

All cost terms are designed for MINIMIZATION (lower = better).
"""

import numpy as np
from typing import Optional, Tuple


def get_box_corners(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box.

    Args:
        center: (3,) [x, y, z]
        size: (3,) [l, w, h]
        yaw: rotation around z-axis

    Returns:
        corners: (8, 3) -- bottom 4 then top 4
    """
    l, w, h = size
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    dx = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    dy = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    dz = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])

    rx = cos_y * dx - sin_y * dy
    ry = sin_y * dx + cos_y * dy

    corners = np.stack([rx + center[0], ry + center[1], dz + center[2]], axis=1)
    return corners


def count_points_inside_box(points: np.ndarray, center: np.ndarray,
                            size: np.ndarray, yaw: float) -> int:
    """Count number of points inside a rotated 3D box."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    half_size = size / 2.0

    local = points[:, :3] - center
    rot = np.array([[cos_y, sin_y, 0],
                    [-sin_y, cos_y, 0],
                    [0, 0, 1]])
    rotated = (rot @ local.T).T

    inside = (
        (np.abs(rotated[:, 0]) <= half_size[0]) &
        (np.abs(rotated[:, 1]) <= half_size[1]) &
        (np.abs(rotated[:, 2]) <= half_size[2])
    )
    return int(inside.sum())


def get_inside_mask(points: np.ndarray, center: np.ndarray,
                    size: np.ndarray, yaw: float) -> np.ndarray:
    """Return boolean mask of points inside the box."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    half_size = size / 2.0

    local = points[:, :3] - center
    rot = np.array([[cos_y, sin_y, 0],
                    [-sin_y, cos_y, 0],
                    [0, 0, 1]])
    rotated = (rot @ local.T).T

    inside = (
        (np.abs(rotated[:, 0]) <= half_size[0]) &
        (np.abs(rotated[:, 1]) <= half_size[1]) &
        (np.abs(rotated[:, 2]) <= half_size[2])
    )
    return inside


def j_density(center: np.ndarray, size: np.ndarray, yaw: float,
              points: np.ndarray) -> float:
    """
    Density cost (Eq. 3-4): -Ginside / |Pobj|
    Minimizing this maximizes the proportion of points inside the box.
    """
    n_total = len(points)
    if n_total == 0:
        return 0.0

    n_inside = count_points_inside_box(points, center, size, yaw)
    return -n_inside / n_total


def _point_to_segment_distance(points: np.ndarray, seg_a: np.ndarray,
                               seg_b: np.ndarray) -> np.ndarray:
    """
    Compute minimum distance from each point to a line segment [seg_a, seg_b].

    Args:
        points: (N, 3)
        seg_a: (3,) segment start
        seg_b: (3,) segment end

    Returns:
        distances: (N,)
    """
    ab = seg_b - seg_a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-12:
        return np.linalg.norm(points - seg_a, axis=1)

    ap = points - seg_a
    t = np.clip(ap @ ab / ab_len_sq, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def j_lshape(center: np.ndarray, size: np.ndarray, yaw: float,
             points: np.ndarray, ego_pos: np.ndarray) -> float:
    """
    L-shape edge alignment cost (Eq. 5).

    1. Get 8 box corners, take top 4 (indices 4-7)
    2. Form 4 top edges
    3. Select 2 edges closest to ego
    4. For inside points, compute min distance to nearest of the 2 edges
    5. Sum and normalize by Ginside
    """
    n_total = len(points)
    if n_total == 0:
        return 0.0

    inside_mask = get_inside_mask(points, center, size, yaw)
    n_inside = int(inside_mask.sum())
    if n_inside == 0:
        return 1.0

    corners = get_box_corners(center, size, yaw)
    top = corners[4:8]

    edges = [(top[0], top[1]), (top[1], top[2]),
             (top[2], top[3]), (top[3], top[0])]

    edge_mids = np.array([(a + b) / 2 for a, b in edges])

    ego_xy = ego_pos[:2]
    dists_to_ego = np.sqrt(np.sum((edge_mids[:, :2] - ego_xy) ** 2, axis=1))

    closest_2_idx = np.argsort(dists_to_ego)[:2]

    inside_points = points[inside_mask, :3]

    d1 = _point_to_segment_distance(inside_points, edges[closest_2_idx[0]][0],
                                    edges[closest_2_idx[0]][1])
    d2 = _point_to_segment_distance(inside_points, edges[closest_2_idx[1]][0],
                                    edges[closest_2_idx[1]][1])
    min_dists = np.minimum(d1, d2)

    return min_dists.sum() / n_inside


def j_surface(center: np.ndarray, ego_pos: np.ndarray,
              c_surface: float) -> float:
    """
    Surface pushing cost (Eq. 6): -min(||(tx,ty)-(ex,ey)||, Csurface)
    Pushes box center away from ego (sensor origin).
    """
    dist_xy = np.sqrt((center[0] - ego_pos[0])**2 + (center[1] - ego_pos[1])**2)
    return -min(dist_xy, c_surface)


def project_3d_bbox_to_2d(center, size, yaw, lidar2cam, intrinsic, image_shape):
    """Project 3D bbox to 2D image plane. Returns [x1, y1, x2, y2] or None."""
    l, w, h = size
    corners_local = np.array([
        [-l/2, -w/2, -h/2], [-l/2, -w/2,  h/2],
        [-l/2,  w/2, -h/2], [-l/2,  w/2,  h/2],
        [ l/2, -w/2, -h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2, -h/2], [ l/2,  w/2,  h/2],
    ])

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y, 0],
                    [sin_y,  cos_y, 0],
                    [0, 0, 1]])
    corners_world = (rot @ corners_local.T).T + center

    corners_homo = np.hstack([corners_world, np.ones((8, 1))])
    corners_cam = (lidar2cam @ corners_homo.T).T

    if corners_cam[:, 2].min() <= 0.1:
        return None

    corners_2d = (intrinsic @ corners_cam[:, :3].T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    H, W = image_shape
    x_min = max(0, int(corners_2d[:, 0].min()))
    y_min = max(0, int(corners_2d[:, 1].min()))
    x_max = min(W, int(corners_2d[:, 0].max()))
    y_max = min(H, int(corners_2d[:, 1].max()))

    if x_max <= x_min or y_max <= y_min:
        return None

    return np.array([x_min, y_min, x_max, y_max])


def compute_iou_2d(bbox1, bbox2):
    """Compute 2D IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-6)


def j_2d(center, size, yaw, bbox_2d_gt, lidar2cam, intrinsic,
         image_shape, gamma):
    """
    2D projection IoU cost (Eq. 7): gamma * (1 - IoU)
    """
    proj = project_3d_bbox_to_2d(center, size, yaw, lidar2cam, intrinsic, image_shape)
    if proj is None:
        return gamma

    iou = compute_iou_2d(proj, bbox_2d_gt)
    return gamma * (1.0 - iou)


def total_cost(params: np.ndarray, points: np.ndarray, ego_pos: np.ndarray,
               bbox_2d: np.ndarray, lidar2cam: np.ndarray,
               intrinsic: np.ndarray, image_shape: Tuple[int, int],
               lambda1: float, lambda2: float, lambda3: float,
               gamma: float, c_surface: float) -> float:
    """
    Total cost J(theta) = lambda1*Jdensity + lambda2*Jl-shape + lambda3*Jsurface + J2D

    Args:
        params: (7,) [x, y, z, l, w, h, yaw]
        points: (N, 3) object points
        ego_pos: (3,) ego position in LiDAR frame
        bbox_2d: (4,) [x1, y1, x2, y2] ground truth 2D bbox
        lidar2cam: (4, 4) transformation
        intrinsic: (3, 3) camera intrinsic
        image_shape: (H, W)
        lambda1, lambda2, lambda3, gamma: cost weights
        c_surface: surface distance clipping

    Returns:
        Total cost (lower is better)
    """
    center = params[:3]
    size = params[3:6]
    yaw = params[6]

    size = np.abs(size)
    size = np.maximum(size, 0.1)

    cost_density = lambda1 * j_density(center, size, yaw, points)
    cost_lshape = lambda2 * j_lshape(center, size, yaw, points, ego_pos)
    cost_surface = lambda3 * j_surface(center, ego_pos, c_surface)
    cost_2d = j_2d(center, size, yaw, bbox_2d, lidar2cam, intrinsic,
                   image_shape, gamma)

    return cost_density + cost_lshape + cost_surface + cost_2d


def total_cost_batch(params_batch: np.ndarray, points: np.ndarray,
                     ego_pos: np.ndarray, bbox_2d: np.ndarray,
                     lidar2cam: np.ndarray, intrinsic: np.ndarray,
                     image_shape: Tuple[int, int],
                     lambda1: float, lambda2: float, lambda3: float,
                     gamma: float, c_surface: float) -> np.ndarray:
    """
    Evaluate cost for a batch of particles.

    Args:
        params_batch: (N_swarm, 7) particle positions

    Returns:
        costs: (N_swarm,) cost values
    """
    n = len(params_batch)
    costs = np.full(n, 1e6)
    for i in range(n):
        costs[i] = total_cost(
            params_batch[i], points, ego_pos, bbox_2d,
            lidar2cam, intrinsic, image_shape,
            lambda1, lambda2, lambda3, gamma, c_surface
        )
    return costs
