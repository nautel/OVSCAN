"""
Output formatter for NuScenes evaluation format.

Converts optimization results to NuScenes JSON submission format.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None

from .config import DEFAULT_ATTRIBUTES, get_paths


def transform_lidar_to_global(
    center: np.ndarray,
    yaw: float,
    lidar2ego: np.ndarray,
    ego2global: np.ndarray,
) -> Tuple[List[float], float]:
    """
    Transform bbox from LiDAR frame to global frame.

    NuScenes evaluation expects coordinates in global frame.
    LiDAR -> Ego -> Global

    Args:
        center: [x, y, z] bbox center in LiDAR frame
        yaw: rotation angle (around z-axis) in LiDAR frame (radians)
        lidar2ego: (4, 4) LiDAR to ego vehicle transform
        ego2global: (4, 4) ego vehicle to global transform

    Returns:
        (translation, yaw_global)
    """
    lidar2global = ego2global @ lidar2ego

    center_homo = np.array([center[0], center[1], center[2], 1.0])
    center_global = lidar2global @ center_homo

    rotation = lidar2global[:3, :3]
    yaw_offset = np.arctan2(rotation[1, 0], rotation[0, 0])
    yaw_global = yaw + yaw_offset

    return [float(center_global[0]), float(center_global[1]), float(center_global[2])], float(yaw_global)


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Convert yaw angle (rotation around z-axis) to quaternion (w, x, y, z)."""
    if Quaternion is not None:
        q = Quaternion(axis=[0, 0, 1], angle=yaw)
        return (q.w, q.x, q.y, q.z)
    else:
        w = np.cos(yaw / 2)
        x = 0.0
        y = 0.0
        z = np.sin(yaw / 2)
        return (w, x, y, z)


def size_lwh_to_wlh(size_lwh: np.ndarray) -> List[float]:
    """Convert size from [l, w, h] to NuScenes [w, l, h] format."""
    l, w, h = size_lwh
    return [float(w), float(l), float(h)]


class NuScenesFormatter:
    """
    Formats optimization results for NuScenes evaluation.

    Output format follows NuScenes detection challenge:
    https://www.nuscenes.org/object-detection
    """

    def __init__(
        self,
        modality: str = 'lidar',
        output_dir: Optional[Path] = None,
    ):
        self.modality = modality
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = get_paths()['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_single_detection(
        self,
        result: Dict[str, Any],
        sample_token: str,
        lidar2ego: Optional[np.ndarray] = None,
        ego2global: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Format a single detection for NuScenes submission."""
        center = result['bbox_3d_center']
        size = result['bbox_3d_size']
        yaw = result['bbox_3d_yaw']
        score = result.get('score', 0.5)
        class_name = result.get('class_name', 'car')

        if lidar2ego is not None and ego2global is not None:
            translation, yaw_global = transform_lidar_to_global(
                center, yaw, lidar2ego, ego2global
            )
        else:
            translation = [float(center[0]), float(center[1]), float(center[2])]
            yaw_global = yaw

        quaternion = yaw_to_quaternion(yaw_global)
        size_wlh = size_lwh_to_wlh(size)

        attribute = DEFAULT_ATTRIBUTES.get(class_name, '')

        return {
            'sample_token': sample_token,
            'translation': translation,
            'size': size_wlh,
            'rotation': list(quaternion),
            'velocity': [0.0, 0.0],
            'detection_name': class_name,
            'detection_score': float(score),
            'attribute_name': attribute,
        }

    def format_sample_results(
        self,
        results: List[Dict[str, Any]],
        sample_token: str,
        lidar2ego: Optional[np.ndarray] = None,
        ego2global: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Format all detections for a sample."""
        formatted = []
        for result in results:
            detection = self.format_single_detection(
                result, sample_token, lidar2ego, ego2global
            )
            formatted.append(detection)
        return formatted

    def create_submission_json(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        output_name: Optional[str] = None,
        sample_transforms: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> str:
        """
        Create NuScenes submission JSON file.

        Args:
            all_results: Dict mapping sample_token to list of results
            output_name: Output filename (without extension)
            sample_transforms: Dict mapping sample_token to (lidar2ego, ego2global)

        Returns:
            Path to created JSON file
        """
        if sample_transforms is None:
            print("WARNING: No transforms provided! Coordinates will be in LiDAR frame, "
                  "which will cause mAP=0 in NuScenes evaluation.")
            sample_transforms = {}

        results_dict = {}

        for sample_token, results in all_results.items():
            transforms = sample_transforms.get(sample_token)
            if transforms is not None:
                lidar2ego, ego2global = transforms
            else:
                lidar2ego, ego2global = None, None

            formatted = self.format_sample_results(
                results, sample_token, lidar2ego, ego2global
            )
            results_dict[sample_token] = formatted

        submission = {
            'meta': {
                'use_camera': self.modality == 'camera',
                'use_lidar': self.modality == 'lidar',
                'use_radar': False,
                'use_map': False,
                'use_external': False,
            },
            'results': results_dict,
        }

        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f'results_nusc_{timestamp}'

        output_path = self.output_dir / f'{output_name}.json'

        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)

        print(f"Saved submission to: {output_path}")
        return str(output_path)

    def save_results_pkl(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        output_name: Optional[str] = None,
    ) -> str:
        """Save results as pickle for internal analysis."""
        if output_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_name = f'results_optimized_{timestamp}'

        output_path = self.output_dir / f'{output_name}.pkl'

        serializable_results = {}
        for sample_token, results in all_results.items():
            serializable_results[sample_token] = []
            for r in results:
                r_copy = r.copy()
                for k, v in r_copy.items():
                    if isinstance(v, np.ndarray):
                        r_copy[k] = v.tolist()
                serializable_results[sample_token].append(r_copy)

        with open(output_path, 'wb') as f:
            pickle.dump(serializable_results, f)

        print(f"Saved PKL results to: {output_path}")
        return str(output_path)


def format_results_for_nuscenes(
    all_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Optional[Path] = None,
    output_name: Optional[str] = None,
    save_json: bool = True,
    save_pkl: bool = True,
    sample_transforms: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, str]:
    """
    Convenience function to format and save results.

    Args:
        all_results: Dict mapping sample_token to list of results
        output_dir: Output directory
        output_name: Base name for output files
        save_json: Save NuScenes JSON format
        save_pkl: Save PKL format
        sample_transforms: Dict mapping sample_token to (lidar2ego, ego2global)

    Returns:
        Dict with paths to saved files
    """
    formatter = NuScenesFormatter(output_dir=output_dir)

    output_paths = {}

    if save_json:
        json_path = formatter.create_submission_json(
            all_results, output_name, sample_transforms
        )
        output_paths['json'] = json_path

    if save_pkl:
        pkl_path = formatter.save_results_pkl(all_results, output_name)
        output_paths['pkl'] = pkl_path

    return output_paths
