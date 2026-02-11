"""
Mask processor for extracting LiDAR points from SAM3 masks.

Functions:
- Project LiDAR points to camera image
- Extract points within mask
- Apply point denoising (DBSCAN depth clustering)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .config import (
    VOXEL_SIZE, DEPTH_RANGE,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES, MIN_CLUSTER_SIZE,
    DEPTH_THRESHOLDS, MIN_POINTS_THRESHOLD,
)
from .point_clustering import filter_multiple_objects_by_depth


def project_points_to_camera(
    points: np.ndarray,
    lidar2cam: np.ndarray,
    intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D LiDAR points to 2D camera coordinates.

    Args:
        points: (N, 3) LiDAR points [x, y, z]
        lidar2cam: (4, 4) LiDAR to camera transform
        intrinsic: (3, 3) or (3, 4) camera intrinsic matrix

    Returns:
        points_2d: (M, 2) projected pixel coordinates for points in front of camera
        in_front_mask: (N,) boolean mask of points in front of camera
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.zeros(0, dtype=bool)

    points_homo = np.hstack([points[:, :3], np.ones((len(points), 1))])
    points_cam = (lidar2cam @ points_homo.T).T

    in_front_mask = points_cam[:, 2] > 0
    points_cam_valid = points_cam[in_front_mask]

    if len(points_cam_valid) == 0:
        return np.array([]).reshape(0, 2), in_front_mask

    if intrinsic.shape == (3, 3):
        K = intrinsic
    else:
        K = intrinsic[:, :3]

    points_img = (K @ points_cam_valid[:, :3].T).T
    points_2d = points_img[:, :2] / points_img[:, 2:3]

    return points_2d, in_front_mask


def get_points_in_mask(
    lidar_points: np.ndarray,
    sam_mask: np.ndarray,
    lidar2cam: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get LiDAR points that project into 2D mask.

    Args:
        lidar_points: (N, 3) LiDAR points
        sam_mask: (H, W) instance mask (>0 means inside)
        lidar2cam: (4, 4) transform
        intrinsic: (3, 3) camera matrix
        image_shape: (H, W)

    Returns:
        in_mask_points: (M, 3) points inside mask
        out_mask_points: (K, 3) points outside mask
        valid_indices: indices of in_mask_points in original array
    """
    H, W = image_shape

    points_2d, in_front_mask = project_points_to_camera(
        lidar_points, lidar2cam, intrinsic
    )

    if len(points_2d) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([], dtype=int)

    front_indices = np.where(in_front_mask)[0]
    front_points = lidar_points[in_front_mask]

    in_image = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    )

    visible_indices = front_indices[in_image]
    visible_points = front_points[in_image]
    visible_2d = points_2d[in_image]

    if len(visible_points) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), np.array([], dtype=int)

    u = visible_2d[:, 0].astype(int)
    v = visible_2d[:, 1].astype(int)

    if len(sam_mask.shape) == 3:
        sam_mask = sam_mask[0]

    in_mask = sam_mask[v, u] > 0

    in_mask_points = visible_points[in_mask]
    out_mask_points = visible_points[~in_mask]
    valid_indices = visible_indices[in_mask]

    return in_mask_points, out_mask_points, valid_indices


class MaskProcessor:
    """
    Processor for extracting objects from SAM3 masks and preparing for optimization.

    Handles:
    - Extracting LiDAR points from masks
    - Applying depth clustering (DBSCAN)
    """

    def __init__(
        self,
        voxel_size: Tuple[float, float, float] = VOXEL_SIZE,
        depth_range: Tuple[float, float] = DEPTH_RANGE,
        use_depth_clustering: bool = True,
    ):
        self.voxel_size = voxel_size
        self.depth_range = depth_range
        self.use_depth_clustering = use_depth_clustering

    def extract_objects_from_masks(
        self,
        lidar_points: np.ndarray,
        mask: np.ndarray,
        metadata: Dict,
        camera_info: Dict,
    ) -> List[Dict[str, Any]]:
        """
        Extract objects from a single camera's mask.

        Args:
            lidar_points: (N, 3) LiDAR points
            mask: (H, W) instance mask (values are instance IDs)
            metadata: Dict with 'instances' list
            camera_info: Dict with 'lidar2cam', 'cam2img'

        Returns:
            List of object dicts
        """
        objects = []

        lidar2cam = np.array(camera_info.get('lidar2cam', np.eye(4)))
        intrinsic = np.array(camera_info.get('cam2img', np.eye(3)))
        image_shape = camera_info.get('img_shape', (900, 1600))[:2]

        instances = metadata.get('instances', [])

        for inst in instances:
            instance_id = inst['instance_id']
            class_name = inst['class_name']
            score = inst['score']
            bbox_2d = np.array(inst['bbox'])

            instance_mask = (mask == instance_id).astype(np.int16)

            points_in_mask, _, _ = get_points_in_mask(
                lidar_points, instance_mask, lidar2cam, intrinsic, image_shape
            )

            min_pts = MIN_POINTS_THRESHOLD.get(class_name, 5)
            if len(points_in_mask) < min_pts:
                continue

            objects.append({
                'instance_id': instance_id,
                'class_name': class_name,
                'label_text': class_name,
                'score': score,
                'bbox_2d': bbox_2d,
                'points': points_in_mask,
                'mask_area': inst.get('mask_area', 0),
                'lidar2cam': lidar2cam,
                'intrinsic': intrinsic,
                'image_shape': image_shape,
            })

        return objects

    def apply_depth_clustering(
        self,
        objects: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """Apply DBSCAN depth clustering to filter noisy points."""
        if not self.use_depth_clustering:
            for obj in objects:
                obj['filtered_points'] = obj['points']
                obj['filtered_point_count'] = len(obj['points'])
                obj['depth_filter_applied'] = False
            return objects

        objects = filter_multiple_objects_by_depth(
            all_objects=objects,
            depth_thresholds=DEPTH_THRESHOLDS,
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            min_cluster_size=MIN_CLUSTER_SIZE,
            verbose=verbose,
            use_adaptive_eps=True,
        )

        return objects

    def process_camera_detections(
        self,
        lidar_points: np.ndarray,
        mask: np.ndarray,
        metadata: Dict,
        camera_info: Dict,
        camera_name: str,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Full processing pipeline for a single camera.

        Steps:
        1. Extract objects from masks
        2. Apply DBSCAN depth clustering
        3. Filter objects with too few points

        Returns:
            List of processed object dicts ready for optimization
        """
        objects = self.extract_objects_from_masks(
            lidar_points, mask, metadata, camera_info
        )

        if len(objects) == 0:
            return []

        for obj in objects:
            obj['camera'] = camera_name

        objects = self.apply_depth_clustering(objects, verbose=verbose)

        filtered_objects = []
        for obj in objects:
            filtered_points = obj.get('filtered_points', obj['points'])
            min_pts = MIN_POINTS_THRESHOLD.get(obj['class_name'], 5)
            if len(filtered_points) >= min_pts:
                filtered_objects.append(obj)

        return filtered_objects


def process_all_cameras(
    processor: MaskProcessor,
    lidar_points: np.ndarray,
    cameras_data: Dict[str, Dict],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process all 6 cameras and combine results.

    Args:
        processor: MaskProcessor instance
        lidar_points: (N, 3) LiDAR points
        cameras_data: Dict mapping camera name to mask/metadata/camera_info
        verbose: Print debug info

    Returns:
        Combined list of objects from all cameras
    """
    all_objects = []

    for camera_name, cam_data in cameras_data.items():
        mask = cam_data.get('mask')
        metadata = cam_data.get('metadata', {})
        camera_info = cam_data.get('camera_info', {})

        if mask is None or camera_info is None:
            continue

        if 'img_shape' not in camera_info:
            camera_info['img_shape'] = mask.shape[:2]

        objects = processor.process_camera_detections(
            lidar_points=lidar_points,
            mask=mask,
            metadata=metadata,
            camera_info=camera_info,
            camera_name=camera_name,
            verbose=verbose,
        )

        all_objects.extend(objects)

        if verbose:
            print(f"  {camera_name}: {len(objects)} objects")

    return all_objects
