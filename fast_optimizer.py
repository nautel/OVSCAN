"""
Ultra-fast 3D bounding box optimizer using geometric methods.

Target: <100ms per object optimization.

Techniques:
- PCA/ConvexHull for O(n) yaw estimation
- Numba JIT compilation for point-in-box tests
- Mini grid search (72 candidates) instead of PSO iterations
- Class-specific strategies
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from .config import ANCHOR_SIZES, MIN_POINTS_THRESHOLD


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _angle_normalize(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


# =============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _count_points_in_box_numba(
        points: np.ndarray,
        center: np.ndarray,
        half_size: np.ndarray,
        cos_yaw: float,
        sin_yaw: float,
    ) -> int:
        """Numba-accelerated point-in-box counting."""
        count = 0
        for i in range(len(points)):
            dx = points[i, 0] - center[0]
            dy = points[i, 1] - center[1]
            dz = points[i, 2] - center[2]

            rx = dx * cos_yaw + dy * sin_yaw
            ry = -dx * sin_yaw + dy * cos_yaw
            rz = dz

            if (abs(rx) <= half_size[0] and
                abs(ry) <= half_size[1] and
                abs(rz) <= half_size[2]):
                count += 1

        return count

    @jit(nopython=True, cache=True)
    def _compute_surface_distances_numba(
        points: np.ndarray,
        center: np.ndarray,
        half_size: np.ndarray,
        cos_yaw: float,
        sin_yaw: float,
    ) -> np.ndarray:
        """Compute distances from points to nearest box surface."""
        n = len(points)
        distances = np.empty(n, dtype=np.float64)

        for i in range(n):
            dx = points[i, 0] - center[0]
            dy = points[i, 1] - center[1]
            dz = points[i, 2] - center[2]

            rx = dx * cos_yaw + dy * sin_yaw
            ry = -dx * sin_yaw + dy * cos_yaw
            rz = dz

            d_x_pos = half_size[0] - rx
            d_x_neg = half_size[0] + rx
            d_y_pos = half_size[1] - ry
            d_y_neg = half_size[1] + ry
            d_z_pos = half_size[2] - rz
            d_z_neg = half_size[2] + rz

            min_d = min(d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg)
            distances[i] = min_d

        return distances

else:
    def _count_points_in_box_numba(points, center, half_size, cos_yaw, sin_yaw):
        """Fallback numpy implementation."""
        points_local = points[:, :3] - center

        rot_matrix = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ])
        points_rotated = (rot_matrix @ points_local.T).T

        inside = (
            (np.abs(points_rotated[:, 0]) <= half_size[0]) &
            (np.abs(points_rotated[:, 1]) <= half_size[1]) &
            (np.abs(points_rotated[:, 2]) <= half_size[2])
        )
        return inside.sum()

    def _compute_surface_distances_numba(points, center, half_size, cos_yaw, sin_yaw):
        """Fallback numpy implementation."""
        points_local = points[:, :3] - center

        rot_matrix = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1],
        ])
        points_rotated = (rot_matrix @ points_local.T).T

        d_x_pos = half_size[0] - points_rotated[:, 0]
        d_x_neg = half_size[0] + points_rotated[:, 0]
        d_y_pos = half_size[1] - points_rotated[:, 1]
        d_y_neg = half_size[1] + points_rotated[:, 1]
        d_z_pos = half_size[2] - points_rotated[:, 2]
        d_z_neg = half_size[2] + points_rotated[:, 2]

        min_d = np.minimum.reduce([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        return min_d


def count_points_in_box_fast(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> int:
    """Fast point-in-box counting with Numba acceleration."""
    if len(points) == 0:
        return 0

    half_size = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    return _count_points_in_box_numba(
        points[:, :3].astype(np.float64),
        center.astype(np.float64),
        half_size.astype(np.float64),
        cos_yaw, sin_yaw,
    )


def compute_surface_score_fast(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    threshold: float = 0.15,
) -> float:
    """Compute surface alignment score."""
    if len(points) == 0:
        return 0.0

    half_size = np.array([size[0] / 2, size[1] / 2, size[2] / 2])
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    distances = _compute_surface_distances_numba(
        points[:, :3].astype(np.float64),
        center.astype(np.float64),
        half_size.astype(np.float64),
        cos_yaw, sin_yaw,
    )

    inside_mask = distances >= 0
    surface_mask = (distances >= 0) & (distances <= threshold)

    if inside_mask.sum() == 0:
        return 0.0

    return surface_mask.sum() / len(points)


def compute_center_alignment_score(
    predicted_center: np.ndarray,
    points: np.ndarray,
    size: np.ndarray,
) -> float:
    """Compute center alignment score between predicted center and point distribution."""
    if len(points) < 3:
        return 0.5

    point_centroid = np.median(points[:, :3], axis=0)

    max_offset = min(size[0], size[1]) * 0.5
    max_offset = max(max_offset, 0.5)

    offset_xy = np.linalg.norm(predicted_center[:2] - point_centroid[:2])
    offset_z = abs(predicted_center[2] - point_centroid[2])

    combined_offset = offset_xy + 0.3 * offset_z

    score = max(0.0, 1.0 - combined_offset / (2 * max_offset))
    return score


# =============================================================================
# YAW ESTIMATION FUNCTIONS
# =============================================================================

def estimate_yaw_pca(points: np.ndarray) -> float:
    """Estimate yaw using PCA on XY plane."""
    if len(points) < 3:
        return 0.0

    points_2d = points[:, :2]

    try:
        pca = PCA(n_components=min(2, len(points_2d)))
        pca.fit(points_2d)
        yaw_vec = pca.components_[0]
        return np.arctan2(yaw_vec[1], yaw_vec[0])
    except Exception:
        return 0.0


def estimate_yaw_convex_hull(points: np.ndarray) -> float:
    """Estimate yaw using minimum area bounding rectangle from convex hull."""
    if len(points) < 4:
        return estimate_yaw_pca(points)

    points_2d = points[:, :2]

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        min_area = float('inf')
        best_yaw = 0.0

        for i in range(len(hull_points)):
            edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
            yaw = np.arctan2(edge[1], edge[0])

            cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
            rotated_x = points_2d[:, 0] * cos_yaw - points_2d[:, 1] * sin_yaw
            rotated_y = points_2d[:, 0] * sin_yaw + points_2d[:, 1] * cos_yaw

            area = (rotated_x.max() - rotated_x.min()) * (rotated_y.max() - rotated_y.min())

            if area < min_area:
                min_area = area
                best_yaw = yaw

        return best_yaw

    except Exception:
        return estimate_yaw_pca(points)


def estimate_yaw_hybrid(points: np.ndarray, size: np.ndarray) -> float:
    """Hybrid yaw estimation: ConvexHull for elongated, PCA for compact."""
    aspect_ratio = size[0] / max(size[1], 0.01)

    if aspect_ratio > 1.5 and len(points) >= 8:
        return estimate_yaw_convex_hull(points)
    else:
        return estimate_yaw_pca(points)


def refine_yaw_multi_angle(
    base_yaw: float,
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    n_angles: int = 8,
) -> float:
    """Refine yaw by testing multiple angles around base estimate."""
    if len(points) < 3:
        return base_yaw

    best_yaw = base_yaw
    best_score = -1

    angles = [base_yaw + (np.pi * i / n_angles) for i in range(n_angles)]

    for yaw in angles:
        score = compute_surface_score_fast(points, center, size, yaw)
        if score > best_score:
            best_score = score
            best_yaw = yaw

    return best_yaw


def estimate_yaw_360_with_2d_projection(
    points: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    bbox_2d_target: np.ndarray,
    lidar2cam: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: Tuple[int, int],
    sam_mask: Optional[np.ndarray] = None,
    n_candidates: int = 8,
) -> float:
    """Full 360 yaw search using 2D reprojection IoU scoring."""
    if len(points) < 3:
        return 0.0

    yaw_candidates = np.linspace(0, 2 * np.pi, n_candidates, endpoint=False)

    best_yaw = 0.0
    best_score = -1.0

    for yaw in yaw_candidates:
        bbox_2d_proj = project_3d_bbox_to_2d(
            center, size, yaw, lidar2cam, intrinsic, image_shape
        )

        if bbox_2d_proj is None:
            continue

        if sam_mask is not None:
            score = compute_iou_with_mask(bbox_2d_proj, sam_mask)
        else:
            score = compute_iou_2d(bbox_2d_proj, bbox_2d_target)

        if score > best_score:
            best_score = score
            best_yaw = yaw

    return best_yaw


def disambiguate_pca_direction(
    points: np.ndarray,
    yaw_pca: float,
    lidar2cam: np.ndarray,
) -> float:
    """Disambiguate PCA direction by checking which direction faces the camera."""
    if len(points) < 3:
        return yaw_pca

    centroid = np.mean(points[:, :3], axis=0)

    cam2lidar = np.linalg.inv(lidar2cam)
    camera_pos_lidar = cam2lidar[:3, 3]

    to_camera = camera_pos_lidar - centroid
    to_camera_angle = np.arctan2(to_camera[1], to_camera[0])

    yaw_options = [yaw_pca, yaw_pca + np.pi]

    def normalize_angle(a):
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

    diffs = [abs(normalize_angle(y - to_camera_angle)) for y in yaw_options]
    return yaw_options[np.argmin(diffs)]


# =============================================================================
# CENTER ESTIMATION
# =============================================================================

def estimate_center_median(points: np.ndarray) -> np.ndarray:
    """Estimate center using median (robust to outliers)."""
    if len(points) == 0:
        return np.array([0.0, 0.0, 0.0])
    return np.median(points[:, :3], axis=0)


def estimate_center_smart(
    points: np.ndarray,
    voxels: np.ndarray,
    size: np.ndarray,
    yaw: float,
) -> np.ndarray:
    """Smart center estimation using point distribution with skewness correction."""
    center = estimate_center_median(voxels if len(voxels) > 0 else points)

    if len(points) < 5:
        return center

    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
    rel_points = points[:, :2] - center[:2]

    rx = rel_points[:, 0] * cos_yaw + rel_points[:, 1] * sin_yaw
    ry = -rel_points[:, 0] * sin_yaw + rel_points[:, 1] * cos_yaw

    mean_rx = np.mean(rx)
    mean_ry = np.mean(ry)

    correction_scale = 0.25
    dx = -mean_rx * correction_scale
    dy = -mean_ry * correction_scale

    center[0] += dx * np.cos(yaw) - dy * np.sin(yaw)
    center[1] += dx * np.sin(yaw) + dy * np.cos(yaw)

    return center


# =============================================================================
# GROUND ESTIMATION
# =============================================================================

def estimate_ground_z_from_points(
    lidar_points: np.ndarray,
    x_center: float,
    y_center: float,
    search_radius: float = 5.0,
    default_ground_z: float = -1.7,
) -> float:
    """Estimate ground Z level from nearby LiDAR points."""
    if len(lidar_points) == 0:
        return default_ground_z

    distances_xy = np.sqrt(
        (lidar_points[:, 0] - x_center)**2 +
        (lidar_points[:, 1] - y_center)**2
    )
    nearby_mask = distances_xy < search_radius
    nearby_points = lidar_points[nearby_mask]

    if len(nearby_points) < 10:
        nearby_mask = distances_xy < search_radius * 2
        nearby_points = lidar_points[nearby_mask]

    if len(nearby_points) < 5:
        return default_ground_z

    z_values = nearby_points[:, 2]
    ground_z = np.percentile(z_values, 10)

    if ground_z > 0 or ground_z < -3.0:
        return default_ground_z

    return ground_z


# =============================================================================
# DEPTH ESTIMATION FOR SMALL OBJECTS
# =============================================================================

def estimate_depth_for_small_object(
    bbox_2d: np.ndarray,
    lidar_points: np.ndarray,
    lidar2cam: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: Tuple[int, int],
    search_radius_pixels: int = 50,
    verbose: bool = False,
) -> Optional[float]:
    """Estimate depth for small objects by finding nearest LiDAR points around the 2D bbox center."""
    if len(lidar_points) == 0:
        return None

    cx = (bbox_2d[0] + bbox_2d[2]) / 2
    cy = (bbox_2d[1] + bbox_2d[3]) / 2
    bbox_width = bbox_2d[2] - bbox_2d[0]
    bbox_height = bbox_2d[3] - bbox_2d[1]

    points_3d = lidar_points[:, :3]
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    points_cam = (lidar2cam @ points_homo.T).T[:, :3]

    valid_depth = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_depth]

    if len(points_cam) == 0:
        return None

    points_img = (intrinsic @ points_cam.T).T
    points_2d = points_img[:, :2] / points_img[:, 2:3]

    h, w = image_shape
    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    points_2d = points_2d[in_image]
    points_cam = points_cam[in_image]

    if len(points_2d) == 0:
        return None

    dist_to_center = np.sqrt((points_2d[:, 0] - cx)**2 + (points_2d[:, 1] - cy)**2)

    nearby_mask = dist_to_center < search_radius_pixels
    if nearby_mask.sum() < 3:
        nearby_mask = dist_to_center < search_radius_pixels * 2

    if nearby_mask.sum() < 3:
        expand_factor = 1.5
        in_expanded_bbox = (
            (points_2d[:, 0] >= bbox_2d[0] - bbox_width * expand_factor) &
            (points_2d[:, 0] <= bbox_2d[2] + bbox_width * expand_factor) &
            (points_2d[:, 1] >= bbox_2d[1] - bbox_height * expand_factor) &
            (points_2d[:, 1] <= bbox_2d[3] + bbox_height * expand_factor)
        )
        nearby_mask = in_expanded_bbox

    if nearby_mask.sum() == 0:
        return None

    nearby_depths = points_cam[nearby_mask, 2]
    estimated_depth = np.median(nearby_depths)

    if verbose:
        print(f"  Depth estimation: found {nearby_mask.sum()} nearby points, "
              f"median depth = {estimated_depth:.2f}m")

    return estimated_depth


def create_bbox_from_depth_estimate(
    bbox_2d: np.ndarray,
    estimated_depth: float,
    class_name: str,
    intrinsic: np.ndarray,
    lidar2cam: np.ndarray,
    detection_score: float = 0.5,
    verbose: bool = False,
    lidar_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Create 3D bbox for small objects using estimated depth."""
    cx = (bbox_2d[0] + bbox_2d[2]) / 2
    cy = bbox_2d[1] + (bbox_2d[3] - bbox_2d[1]) * 0.7

    anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['pedestrian'])
    anchor_size = np.array(anchors[1])

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx_int = intrinsic[0, 2]
    cy_int = intrinsic[1, 2]

    x_cam = (cx - cx_int) * estimated_depth / fx
    y_cam = (cy - cy_int) * estimated_depth / fy
    z_cam = estimated_depth

    cam2lidar = np.linalg.inv(lidar2cam)
    point_cam = np.array([x_cam, y_cam, z_cam, 1])
    point_lidar = cam2lidar @ point_cam

    center = point_lidar[:3].copy()

    object_height = anchor_size[2]

    if lidar_points is not None and len(lidar_points) > 0:
        ground_z = estimate_ground_z_from_points(
            lidar_points=lidar_points,
            x_center=center[0],
            y_center=center[1],
            search_radius=5.0,
            default_ground_z=-1.7,
        )
    else:
        ground_z = -1.7

    center[2] = ground_z + object_height / 2

    yaw = 0.0
    confidence = detection_score * 0.6

    return {
        'bbox_3d_center': center,
        'bbox_3d_size': anchor_size,
        'bbox_3d_yaw': yaw,
        'score': confidence,
        'points_inside': 0,
        'iou_2d': 0.0,
        'bbox_2d_proj': None,
        'method': 'DEPTH_ESTIMATE',
        'score_density': 0,
        'score_surface': 0,
        'score_iou': 0,
        'estimated_depth': estimated_depth,
    }


def compute_initial_bbox_from_points(points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute oriented bounding box from 2D point cloud (bird's eye view)."""
    if len(points_2d) < 3:
        center = np.mean(points_2d, axis=0)
        size = np.ptp(points_2d, axis=0) if len(points_2d) > 1 else np.array([1.0, 1.0])
        return center, size, 0.0

    try:
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
    except Exception:
        center = np.mean(points_2d, axis=0)
        size = np.ptp(points_2d, axis=0)
        return center, np.maximum(size, [0.5, 0.5]), 0.0

    min_area = float('inf')
    best_rect = (np.mean(points_2d, axis=0), np.array([1.0, 1.0]), 0.0)

    for i in range(len(hull_points)):
        edge = hull_points[(i + 1) % len(hull_points)] - hull_points[i]
        angle = np.arctan2(edge[1], edge[0])

        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotated = np.column_stack([
            cos_a * hull_points[:, 0] - sin_a * hull_points[:, 1],
            sin_a * hull_points[:, 0] + cos_a * hull_points[:, 1],
        ])

        min_xy = np.min(rotated, axis=0)
        max_xy = np.max(rotated, axis=0)
        area = (max_xy[0] - min_xy[0]) * (max_xy[1] - min_xy[1])

        if area < min_area and area > 0:
            min_area = area
            center_rot = (min_xy + max_xy) / 2
            size = max_xy - min_xy

            center = np.array([
                cos_a * center_rot[0] + sin_a * center_rot[1],
                -sin_a * center_rot[0] + cos_a * center_rot[1],
            ])
            best_rect = (center, size, angle)

    return best_rect


def compute_lshape_center_offset(
    points: np.ndarray,
    initial_center: np.ndarray,
    yaw: float,
    size: np.ndarray,
) -> Tuple[float, float]:
    """Compute L-shape center offset for vehicles."""
    if len(points) < 5:
        return 0.0, 0.0

    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    local_pts = points[:, :2] - initial_center[:2]
    rotated = np.column_stack([
        cos_y * local_pts[:, 0] - sin_y * local_pts[:, 1],
        sin_y * local_pts[:, 0] + cos_y * local_pts[:, 1],
    ])

    q1 = np.sum((rotated[:, 0] > 0) & (rotated[:, 1] > 0))
    q2 = np.sum((rotated[:, 0] < 0) & (rotated[:, 1] > 0))
    q3 = np.sum((rotated[:, 0] < 0) & (rotated[:, 1] < 0))
    q4 = np.sum((rotated[:, 0] > 0) & (rotated[:, 1] < 0))

    quadrants = [q1, q2, q3, q4]
    total = sum(quadrants)

    if total == 0:
        return 0.0, 0.0

    sorted_quads = sorted(quadrants, reverse=True)
    top_two_ratio = (sorted_quads[0] + sorted_quads[1]) / total

    if top_two_ratio < 0.6:
        return 0.0, 0.0

    dominant = np.argmax(quadrants)

    offset_scale = 0.20
    offsets = [
        (-size[0] / 2 * offset_scale, -size[1] / 2 * offset_scale),
        (+size[0] / 2 * offset_scale, -size[1] / 2 * offset_scale),
        (+size[0] / 2 * offset_scale, +size[1] / 2 * offset_scale),
        (-size[0] / 2 * offset_scale, +size[1] / 2 * offset_scale),
    ]

    dx_local, dy_local = offsets[dominant]

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    world_dx = cos_yaw * dx_local - sin_yaw * dy_local
    world_dy = sin_yaw * dx_local + cos_yaw * dy_local

    return world_dx, world_dy


# =============================================================================
# 2D PROJECTION AND IoU
# =============================================================================

def project_3d_bbox_to_2d(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    lidar2cam: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Project 3D bbox to 2D image plane. Returns [x1, y1, x2, y2] or None."""
    l, w, h = size

    corners_local = np.array([
        [-l/2, -w/2, -h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2, -h/2],
        [-l/2,  w/2,  h/2],
        [ l/2, -w/2, -h/2],
        [ l/2, -w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
    ])

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1],
    ])
    corners_world = (rot @ corners_local.T).T + center

    corners_homo = np.hstack([corners_world, np.ones((8, 1))])
    corners_cam = (lidar2cam @ corners_homo.T).T

    depths = corners_cam[:, 2]
    if depths.min() <= 0.1:
        return None

    corners_2d = (intrinsic @ corners_cam[:, :3].T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]

    x_min = max(0, int(corners_2d[:, 0].min()))
    y_min = max(0, int(corners_2d[:, 1].min()))
    x_max = min(image_shape[1], int(corners_2d[:, 0].max()))
    y_max = min(image_shape[0], int(corners_2d[:, 1].max()))

    if x_max <= x_min or y_max <= y_min:
        return None

    return np.array([x_min, y_min, x_max, y_max])


def compute_iou_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute 2D IoU between two bboxes [x1, y1, x2, y2]."""
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


def compute_iou_with_mask(bbox: np.ndarray, mask: np.ndarray) -> float:
    """Compute IoU between projected bbox and SAM mask."""
    x1, y1, x2, y2 = bbox.astype(int)

    h, w = mask.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask, dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    mask_bool = mask > 0
    intersection = (bbox_mask & mask_bool).sum()
    union = (bbox_mask | mask_bool).sum()

    return intersection / max(union, 1)


# =============================================================================
# ULTRA FAST OPTIMIZER CONFIG
# =============================================================================

@dataclass
class UltraFastConfig:
    """Configuration for UltraFastOptimizer."""
    yaw_method: str = 'hybrid'
    n_yaw_candidates: int = 8
    n_position_offsets: int = 3
    position_delta: float = 0.5
    weight_iou: float = 0.5
    weight_density: float = 0.25
    weight_surface: float = 0.10
    weight_center: float = 0.15
    surface_threshold: float = 0.2
    min_iou: float = 0.1


CLASS_FAST_CONFIGS = {
    'car': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=8, n_position_offsets=3,
        weight_iou=0.35, weight_density=0.30, weight_surface=0.20, weight_center=0.15,
    ),
    'truck': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=12, n_position_offsets=5,
        position_delta=1.2,
        weight_iou=0.25, weight_density=0.35, weight_surface=0.25, weight_center=0.15,
    ),
    'bus': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=16, n_position_offsets=5,
        position_delta=1.5,
        weight_iou=0.25, weight_density=0.35, weight_surface=0.25, weight_center=0.15,
    ),
    'trailer': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=24, n_position_offsets=7,
        position_delta=2.0,
        weight_iou=0.20, weight_density=0.35, weight_surface=0.30, weight_center=0.15,
    ),
    'construction_vehicle': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=8, n_position_offsets=3,
        weight_iou=0.25, weight_density=0.35, weight_surface=0.25, weight_center=0.15,
    ),
    'pedestrian': UltraFastConfig(
        yaw_method='pca', n_yaw_candidates=4, n_position_offsets=3,
        position_delta=0.4,
        weight_iou=0.40, weight_density=0.25, weight_surface=0.20, weight_center=0.15,
    ),
    'motorcycle': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=8, n_position_offsets=3,
        position_delta=0.5,
        weight_iou=0.35, weight_density=0.30, weight_surface=0.20, weight_center=0.15,
    ),
    'bicycle': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=8, n_position_offsets=3,
        position_delta=0.4,
        weight_iou=0.35, weight_density=0.30, weight_surface=0.20, weight_center=0.15,
    ),
    'traffic_cone': UltraFastConfig(
        yaw_method='pca', n_yaw_candidates=2, n_position_offsets=3,
        position_delta=0.3,
        weight_iou=0.40, weight_density=0.25, weight_surface=0.20, weight_center=0.15,
    ),
    'barrier': UltraFastConfig(
        yaw_method='convex_hull', n_yaw_candidates=6, n_position_offsets=2,
        weight_iou=0.35, weight_density=0.30, weight_surface=0.20, weight_center=0.15,
    ),
}


def get_fast_config(class_name: str) -> UltraFastConfig:
    """Get class-specific configuration."""
    return CLASS_FAST_CONFIGS.get(class_name, UltraFastConfig())


# =============================================================================
# ULTRA FAST OPTIMIZER CLASS
# =============================================================================

class UltraFastOptimizer:
    """
    Ultra-fast 3D bounding box optimizer.

    Target performance: <100ms per object.

    Pipeline:
    1. PCA/ConvexHull yaw estimation (~1-3ms)
    2. Smart center estimation (~1ms)
    3. Mini grid search with N_yaw * N_pos candidates (~20-50ms)
    4. Multi-anchor selection (test 3 sizes) (~10-30ms)
    5. Return best result

    Total: 30-80ms typical
    """

    def __init__(
        self,
        try_all_anchors: bool = True,
        verbose: bool = False,
    ):
        self.try_all_anchors = try_all_anchors
        self.verbose = verbose

    def _estimate_yaw(
        self,
        points: np.ndarray,
        size: np.ndarray,
        config: UltraFastConfig,
    ) -> float:
        if config.yaw_method == 'pca':
            return estimate_yaw_pca(points)
        elif config.yaw_method == 'convex_hull':
            return estimate_yaw_convex_hull(points)
        else:
            return estimate_yaw_hybrid(points, size)

    def _generate_candidates(
        self,
        base_center: np.ndarray,
        base_yaw: float,
        config: UltraFastConfig,
    ) -> List[Tuple[np.ndarray, float]]:
        """Generate candidate (center, yaw) pairs for grid search."""
        candidates = []

        n_yaw = config.n_yaw_candidates
        n_pos = config.n_position_offsets
        delta = config.position_delta

        yaw_deltas = np.linspace(-np.pi / 4, np.pi / 4, n_yaw)
        pos_offsets = np.linspace(-delta, delta, 2 * n_pos + 1)

        for yaw_delta in yaw_deltas:
            yaw = base_yaw + yaw_delta

            for dx in pos_offsets:
                for dy in pos_offsets:
                    cos_yaw, sin_yaw = np.cos(base_yaw), np.sin(base_yaw)
                    world_dx = dx * cos_yaw - dy * sin_yaw
                    world_dy = dx * sin_yaw + dy * cos_yaw

                    center = base_center.copy()
                    center[0] += world_dx
                    center[1] += world_dy

                    candidates.append((center, yaw))

        return candidates

    def _evaluate_candidate(
        self,
        center: np.ndarray,
        size: np.ndarray,
        yaw: float,
        points: np.ndarray,
        voxels: np.ndarray,
        bbox_2d_target: np.ndarray,
        lidar2cam: np.ndarray,
        intrinsic: np.ndarray,
        image_shape: Tuple[int, int],
        sam_mask: Optional[np.ndarray],
        config: UltraFastConfig,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate fitness score for a candidate."""
        n_points = count_points_in_box_fast(points, center, size, yaw)

        if n_points == 0:
            bbox_2d_proj = project_3d_bbox_to_2d(
                center, size, yaw, lidar2cam, intrinsic, image_shape
            )
            if bbox_2d_proj is None:
                return -1.0, {}

            if sam_mask is not None:
                iou_score = compute_iou_with_mask(bbox_2d_proj, sam_mask)
            else:
                iou_score = compute_iou_2d(bbox_2d_proj, bbox_2d_target)

            return iou_score * 0.3, {
                'n_points': 0,
                'density_score': 0,
                'iou_score': iou_score,
                'surface_score': 0,
                'center_score': 0,
                'bbox_2d_proj': bbox_2d_proj,
            }

        density_score = n_points / max(len(points), 1)

        bbox_2d_proj = project_3d_bbox_to_2d(
            center, size, yaw, lidar2cam, intrinsic, image_shape
        )

        if bbox_2d_proj is None:
            iou_score = 0.0
        elif sam_mask is not None:
            iou_score = compute_iou_with_mask(bbox_2d_proj, sam_mask)
        else:
            iou_score = compute_iou_2d(bbox_2d_proj, bbox_2d_target)

        surface_score = compute_surface_score_fast(
            points, center, size, yaw, config.surface_threshold
        )

        center_score = compute_center_alignment_score(center, points, size)

        total_score = (
            config.weight_iou * iou_score +
            config.weight_density * density_score +
            config.weight_surface * surface_score +
            config.weight_center * center_score
        )

        details = {
            'n_points': n_points,
            'density_score': density_score,
            'iou_score': iou_score,
            'surface_score': surface_score,
            'center_score': center_score,
            'bbox_2d_proj': bbox_2d_proj,
        }

        return total_score, details

    def optimize_single_anchor(
        self,
        points: np.ndarray,
        voxels: np.ndarray,
        bbox_2d_target: np.ndarray,
        anchor_size: np.ndarray,
        lidar2cam: np.ndarray,
        intrinsic: np.ndarray,
        image_shape: Tuple[int, int],
        class_name: str,
        sam_mask: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Optimize 3D bbox with a single anchor size."""
        config = get_fast_config(class_name)

        if len(points) < 3:
            return None

        vehicle_classes = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']
        is_vehicle = class_name in vehicle_classes

        if len(points) >= 4:
            points_2d = points[:, :2]
            try:
                hull_center, hull_size, hull_yaw = compute_initial_bbox_from_points(points_2d)
                if is_vehicle:
                    base_yaw = hull_yaw
                else:
                    base_yaw = self._estimate_yaw(points, anchor_size, config)
            except Exception:
                base_yaw = self._estimate_yaw(points, anchor_size, config)
                hull_center = None
        else:
            base_yaw = self._estimate_yaw(points, anchor_size, config)
            hull_center = None

        rough_center = estimate_center_median(voxels if len(voxels) > 0 else points)

        large_vehicles = ['trailer', 'truck', 'bus']
        if class_name in large_vehicles:
            base_yaw = estimate_yaw_360_with_2d_projection(
                points=points,
                center=rough_center,
                size=anchor_size,
                bbox_2d_target=bbox_2d_target,
                lidar2cam=lidar2cam,
                intrinsic=intrinsic,
                image_shape=image_shape,
                sam_mask=sam_mask,
                n_candidates=config.n_yaw_candidates,
            )
            base_yaw = disambiguate_pca_direction(points, base_yaw, lidar2cam)
        else:
            base_yaw = refine_yaw_multi_angle(
                base_yaw, points, rough_center, anchor_size,
                n_angles=config.n_yaw_candidates,
            )

        if hull_center is not None and is_vehicle:
            center = np.array([hull_center[0], hull_center[1], np.median(points[:, 2])])
        else:
            center = estimate_center_smart(points, voxels, anchor_size, base_yaw)

        if is_vehicle and len(points) >= 5:
            dx, dy = compute_lshape_center_offset(points, center, base_yaw, anchor_size)
            center[0] += dx
            center[1] += dy

        candidates = self._generate_candidates(center, base_yaw, config)

        best_score = -1
        best_result = None

        for cand_center, cand_yaw in candidates:
            score, details = self._evaluate_candidate(
                cand_center, anchor_size, cand_yaw,
                points, voxels, bbox_2d_target,
                lidar2cam, intrinsic, image_shape,
                sam_mask, config,
            )

            if score > best_score:
                best_score = score
                best_result = {
                    'bbox_3d_center': cand_center.copy(),
                    'bbox_3d_size': anchor_size.copy(),
                    'bbox_3d_yaw': cand_yaw,
                    'score': score,
                    'points_inside': details.get('n_points', 0),
                    'iou_2d': details.get('iou_score', 0),
                    'bbox_2d_proj': details.get('bbox_2d_proj'),
                    'method': 'ULTRA_FAST',
                    'score_density': details.get('density_score', 0),
                    'score_surface': details.get('surface_score', 0),
                    'score_iou': details.get('iou_score', 0),
                }

        return best_result

    def optimize(
        self,
        voxel_centers: np.ndarray,
        points: np.ndarray,
        bbox_2d_target: np.ndarray,
        anchor_size: np.ndarray,
        lidar2cam: np.ndarray,
        intrinsic: np.ndarray,
        image_shape: Tuple[int, int],
        class_name: str = 'car',
        extended_voxels: Optional[np.ndarray] = None,
        sam_mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Main optimization method (compatible with existing interface).
        """
        start_time = time.time()

        voxels = extended_voxels if extended_voxels is not None else voxel_centers
        if len(voxels) == 0:
            voxels = points

        if self.try_all_anchors:
            anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['car'])
            anchor_names = ['min', 'mean', 'max']

            best_result = None
            best_score = -1

            for anchor, name in zip(anchors, anchor_names):
                result = self.optimize_single_anchor(
                    points=points,
                    voxels=voxels,
                    bbox_2d_target=bbox_2d_target,
                    anchor_size=np.array(anchor),
                    lidar2cam=lidar2cam,
                    intrinsic=intrinsic,
                    image_shape=image_shape,
                    class_name=class_name,
                    sam_mask=sam_mask,
                )

                if result is not None and result['score'] > best_score:
                    best_score = result['score']
                    best_result = result
                    best_result['anchor_used'] = name

            elapsed = time.time() - start_time

            if best_result is None:
                center = estimate_center_median(voxels if len(voxels) > 0 else points)
                mean_anchor = np.array(anchors[1])
                yaw = self._estimate_yaw(points, mean_anchor, get_fast_config(class_name))

                best_result = {
                    'bbox_3d_center': center,
                    'bbox_3d_size': mean_anchor,
                    'bbox_3d_yaw': yaw,
                    'score': 0.01,
                    'points_inside': 0,
                    'iou_2d': 0.0,
                    'bbox_2d_proj': None,
                    'method': 'ULTRA_FAST_FALLBACK',
                    'score_density': 0,
                    'score_surface': 0,
                    'score_iou': 0,
                    'anchor_used': 'mean_fallback',
                }

            best_result['optimization_time'] = elapsed

            if self.verbose:
                print(f"  Ultra-fast optimize: {class_name}, "
                      f"score={best_result['score']:.3f}, anchor={best_result.get('anchor_used', 'N/A')}, "
                      f"time={elapsed * 1000:.1f}ms")

            return best_result

        else:
            result = self.optimize_single_anchor(
                points=points,
                voxels=voxels,
                bbox_2d_target=bbox_2d_target,
                anchor_size=anchor_size,
                lidar2cam=lidar2cam,
                intrinsic=intrinsic,
                image_shape=image_shape,
                class_name=class_name,
                sam_mask=sam_mask,
            )

            elapsed = time.time() - start_time

            if result is None:
                center = estimate_center_median(voxels if len(voxels) > 0 else points)
                yaw = self._estimate_yaw(points, anchor_size, get_fast_config(class_name))

                result = {
                    'bbox_3d_center': center,
                    'bbox_3d_size': anchor_size.copy(),
                    'bbox_3d_yaw': yaw,
                    'score': 0.01,
                    'points_inside': 0,
                    'iou_2d': 0.0,
                    'bbox_2d_proj': None,
                    'method': 'ULTRA_FAST_FALLBACK',
                    'score_density': 0,
                    'score_surface': 0,
                    'score_iou': 0,
                }

            result['anchor_used'] = 'provided'
            result['optimization_time'] = elapsed

            return result
