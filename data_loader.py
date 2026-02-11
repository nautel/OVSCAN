"""
Data loader for NuScenes + SAM3 pre-generated masks.

Combines:
- NuScenes info PKL (lidar paths, camera calibration)
- SAM3 mask index and mask files
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .config import CAMERA_NAMES, get_paths


class MaskNuScenesLoader:
    """
    Loader that combines NuScenes data with pre-generated SAM3 masks.

    Attributes:
        data_root: Path to NuScenes data
        sam3_root: Path to SAM3 masks
        split: 'train' or 'val'
        samples: List of sample tokens with mask info
        nuscenes_data: NuScenes info dict
        sample_token_to_info: Mapping from token to NuScenes sample info
    """

    def __init__(
        self,
        split: str = 'train',
        data_root: Optional[str] = None,
        sam3_root: Optional[str] = None,
    ):
        """
        Initialize loader.

        Args:
            split: 'train' or 'val'
            data_root: Path to NuScenes data (overrides env var / default)
            sam3_root: Path to SAM3 masks (overrides env var / default)
        """
        self.split = split
        paths = get_paths(data_root=data_root, sam3_root=sam3_root)
        self.data_root = paths['data_root']
        self.sam3_root = paths['sam3_root']

        # Load NuScenes info
        info_path = paths['nuscenes_info_train'] if split == 'train' else paths['nuscenes_info_val']
        if not info_path.exists():
            raise FileNotFoundError(f"NuScenes info not found: {info_path}")

        with open(info_path, 'rb') as f:
            self.nuscenes_data = pickle.load(f)

        # Create sample_token -> info mapping
        data_list = self.nuscenes_data.get('data_list',
                                           self.nuscenes_data.get('infos', []))
        self.sample_token_to_info = {}
        for sample in data_list:
            token = sample.get('token', sample.get('sample_idx', ''))
            if token:
                self.sample_token_to_info[token] = sample

        # Load SAM3 mask index
        index_path = paths['sam3_index_train'] if split == 'train' else paths['sam3_index_val']
        if not index_path.exists():
            raise FileNotFoundError(f"SAM3 index not found: {index_path}")

        with open(index_path, 'rb') as f:
            self.sam3_index = pickle.load(f)

        self.samples = self.sam3_index.get('samples', [])

        print(f"Loaded {len(self.samples)} samples with SAM3 masks")
        print(f"NuScenes has {len(self.sample_token_to_info)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def get_sample_token(self, idx: int) -> str:
        sample_info = self.samples[idx]
        return sample_info.get('sample_token', '')

    def get_nuscenes_info(self, sample_token: str) -> Optional[Dict]:
        return self.sample_token_to_info.get(sample_token)

    def load_lidar_points(self, sample_token: str) -> np.ndarray:
        """
        Load LiDAR points for a sample.

        Returns:
            points: (N, 3) array of [x, y, z]
        """
        nuscenes_info = self.get_nuscenes_info(sample_token)
        if nuscenes_info is None:
            raise ValueError(f"Sample not found in NuScenes: {sample_token}")

        lidar_info = nuscenes_info.get('lidar_points', {})
        lidar_path = lidar_info.get('lidar_path', '')

        full_path = self.data_root / 'samples' / 'LIDAR_TOP' / lidar_path

        if not full_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {full_path}")

        points = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, 5)
        return points[:, :3]

    def load_mask(self, sample_token: str, camera: str) -> np.ndarray:
        """Load SAM3 mask for a specific camera. Supports .npy and .npz formats.

        Returns (H, W) int16 array.
        """
        mask_dir = self.sam3_root / self.split / 'masks' / sample_token

        # Try .npy first (raw format)
        npy_path = mask_dir / f'{camera}.npy'
        if npy_path.exists():
            return np.load(npy_path)

        # Try .npz (compressed format)
        npz_path = mask_dir / f'{camera}.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            return data['mask']

        return np.zeros((900, 1600), dtype=np.int16)

    def load_metadata(self, sample_token: str, camera: str) -> Dict:
        """Load SAM3 metadata for a specific camera."""
        metadata_path = self.sam3_root / self.split / 'metadata' / sample_token / f'{camera}.json'

        if not metadata_path.exists():
            return {'instances': []}

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def get_camera_info(self, sample_token: str, camera: str) -> Optional[Dict]:
        """Get camera calibration info from NuScenes."""
        nuscenes_info = self.get_nuscenes_info(sample_token)
        if nuscenes_info is None:
            return None

        images = nuscenes_info.get('images', {})

        if camera in images:
            cam_info = images[camera].copy()
            if 'cam2img' in cam_info:
                cam_info['cam2img'] = np.array(cam_info['cam2img'])
            if 'lidar2cam' in cam_info:
                cam_info['lidar2cam'] = np.array(cam_info['lidar2cam'])
            if 'cam2lidar' in cam_info:
                cam_info['cam2lidar'] = np.array(cam_info['cam2lidar'])
            return cam_info

        for cam_key, cam_info in images.items():
            if camera in cam_info.get('img_path', ''):
                result = cam_info.copy()
                if 'cam2img' in result:
                    result['cam2img'] = np.array(result['cam2img'])
                if 'lidar2cam' in result:
                    result['lidar2cam'] = np.array(result['lidar2cam'])
                if 'cam2lidar' in result:
                    result['cam2lidar'] = np.array(result['cam2lidar'])
                return result

        return None

    def get_transforms(self, sample_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lidar2ego and ego2global transforms for a sample.

        Returns:
            (lidar2ego, ego2global) each (4, 4)
        """
        nuscenes_info = self.get_nuscenes_info(sample_token)
        if nuscenes_info is None:
            raise ValueError(f"Sample not found in NuScenes: {sample_token}")

        lidar_info = nuscenes_info.get('lidar_points', {})
        lidar2ego = lidar_info.get('lidar2ego')
        if lidar2ego is None:
            raise ValueError(f"lidar2ego transform not found for sample: {sample_token}")

        ego2global = nuscenes_info.get('ego2global')
        if ego2global is None:
            raise ValueError(f"ego2global transform not found for sample: {sample_token}")

        return np.array(lidar2ego), np.array(ego2global)

    def get_all_cameras_data(self, sample_token: str) -> Dict[str, Dict]:
        """Get mask, metadata, and camera info for all 6 cameras."""
        result = {}
        for camera in CAMERA_NAMES:
            result[camera] = {
                'mask': self.load_mask(sample_token, camera),
                'metadata': self.load_metadata(sample_token, camera),
                'camera_info': self.get_camera_info(sample_token, camera),
            }
        return result

    def get_image_shape(self, camera: str = 'CAM_FRONT') -> Tuple[int, int]:
        """Get image shape (H, W) - typically 900x1600 for NuScenes."""
        return (900, 1600)

    def get_sample_info(self, idx: int) -> Dict:
        return self.samples[idx]


def load_sample_data(loader: MaskNuScenesLoader, idx: int) -> Dict[str, Any]:
    """
    Convenience function to load all data for a sample.

    Returns:
        Dict with sample_token, sample_info, lidar_points, cameras
    """
    sample_info = loader.get_sample_info(idx)
    sample_token = sample_info['sample_token']

    return {
        'sample_token': sample_token,
        'sample_info': sample_info,
        'lidar_points': loader.load_lidar_points(sample_token),
        'cameras': loader.get_all_cameras_data(sample_token),
    }
