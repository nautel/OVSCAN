"""
3D NMS (Non-Maximum Suppression) for multi-camera fusion.

Merges overlapping detections from different cameras based on 3D IoU.
Uses Shapely for rotated BEV IoU with axis-aligned fallback.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .config import NMS_IOU_THRESHOLD, CLASS_NMS_IOU_THRESHOLDS


def get_corners_3d(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """
    Get 8 corners of a 3D bounding box.

    Args:
        center: (3,) [x, y, z]
        size: (3,) [l, w, h]
        yaw: rotation around z-axis in radians

    Returns:
        corners: (8, 3) corner coordinates
    """
    l, w, h = size
    x, y, z = center

    corners_local = np.array([
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ])

    corners = corners_local @ rot_matrix.T + center
    return corners


def compute_iou_3d_rotated(
    center1: np.ndarray, size1: np.ndarray, yaw1: float,
    center2: np.ndarray, size2: np.ndarray, yaw2: float,
) -> float:
    """
    Compute 3D IoU with proper rotation using Shapely polygon intersection for BEV.

    Falls back to axis-aligned approximation if Shapely is not available.
    """
    # Distance pre-filter
    dist_2d = np.linalg.norm(center1[:2] - center2[:2])
    max_diag = max(np.hypot(size1[0], size1[1]), np.hypot(size2[0], size2[1]))
    if dist_2d > max_diag:
        return 0.0

    # Height overlap
    z1_min = center1[2] - size1[2] / 2
    z1_max = center1[2] + size1[2] / 2
    z2_min = center2[2] - size2[2] / 2
    z2_max = center2[2] + size2[2] / 2

    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    z_union = max(z1_max, z2_max) - min(z1_min, z2_min)

    if z_union <= 0:
        return 0.0

    height_iou = z_overlap / z_union

    # BEV IoU
    try:
        from shapely.geometry import Polygon as ShapelyPolygon

        corners1 = get_corners_3d(center1, size1, yaw1)[:4, :2]
        corners2 = get_corners_3d(center2, size2, yaw2)[:4, :2]

        poly1 = ShapelyPolygon(corners1)
        poly2 = ShapelyPolygon(corners2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area

        if union_area <= 0:
            return 0.0

        bev_iou = inter_area / union_area
    except ImportError:
        # Axis-aligned fallback
        corners1 = get_corners_3d(center1, size1, yaw1)[:4, :2]
        corners2 = get_corners_3d(center2, size2, yaw2)[:4, :2]

        min1_bev = corners1.min(axis=0)
        max1_bev = corners1.max(axis=0)
        min2_bev = corners2.min(axis=0)
        max2_bev = corners2.max(axis=0)

        inter_min = np.maximum(min1_bev, min2_bev)
        inter_max = np.minimum(max1_bev, max2_bev)

        if np.any(inter_max <= inter_min):
            return 0.0

        inter_area = (inter_max[0] - inter_min[0]) * (inter_max[1] - inter_min[1])
        area1 = size1[0] * size1[1]
        area2 = size2[0] * size2[1]
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        bev_iou = inter_area / union_area

    return bev_iou * height_iou


def nms_3d_bboxes(
    results: List[Dict[str, Any]],
    iou_threshold: float = NMS_IOU_THRESHOLD,
    class_agnostic: bool = False,
    use_class_specific_threshold: bool = True,
) -> List[Dict[str, Any]]:
    """
    Apply 3D NMS to merge overlapping detections from multiple cameras.

    Args:
        results: List of optimization results with bbox_3d_center, bbox_3d_size,
                 bbox_3d_yaw, score, class_name
        iou_threshold: Default IoU threshold for merging
        class_agnostic: If True, merge across classes
        use_class_specific_threshold: If True, use per-class thresholds

    Returns:
        Filtered results after NMS
    """
    if len(results) == 0:
        return []

    results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)

    keep = []
    suppressed = set()

    for i, res_i in enumerate(results):
        if i in suppressed:
            continue

        keep.append(res_i)

        center_i = np.array(res_i['bbox_3d_center'])
        size_i = np.array(res_i['bbox_3d_size'])
        yaw_i = res_i['bbox_3d_yaw']
        class_i = res_i.get('class_name', 'unknown')

        if use_class_specific_threshold:
            threshold_i = CLASS_NMS_IOU_THRESHOLDS.get(class_i, iou_threshold)
        else:
            threshold_i = iou_threshold

        for j in range(i + 1, len(results)):
            if j in suppressed:
                continue

            res_j = results[j]
            class_j = res_j.get('class_name', 'unknown')

            if not class_agnostic and class_i != class_j:
                continue

            center_j = np.array(res_j['bbox_3d_center'])
            size_j = np.array(res_j['bbox_3d_size'])
            yaw_j = res_j['bbox_3d_yaw']

            iou = compute_iou_3d_rotated(
                center_i, size_i, yaw_i,
                center_j, size_j, yaw_j,
            )

            if iou >= threshold_i:
                suppressed.add(j)

    return keep


def merge_multi_camera_detections(
    results_per_camera: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Merge detections from multiple cameras using 3D NMS.

    Args:
        results_per_camera: Dict mapping camera name to list of results
        iou_threshold: IoU threshold for merging

    Returns:
        Merged results after NMS
    """
    all_results = []
    for camera, results in results_per_camera.items():
        for r in results:
            r['camera'] = camera
            all_results.append(r)

    return nms_3d_bboxes(all_results, iou_threshold=iou_threshold)
