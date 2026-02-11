"""
Configuration for OVSCAN 3D Bounding Box Optimization.

Merged from OPTIMIZATION_BBOX_BASED_ON_MASK/config.py and OPTIMAL_BBOX/config.py.
All paths are configurable via environment variables or function arguments.
"""

import os
from pathlib import Path
from typing import Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_project_root() -> Path:
    """Get project root, defaults to two levels up from this file."""
    return Path(os.environ.get(
        'OVSCAN_PROJECT_ROOT',
        str(Path(__file__).parent.parent)
    ))


def get_paths(
    data_root: Optional[str] = None,
    sam3_root: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Get all paths with env-var fallbacks.

    Priority: explicit arg > env var > default relative to project root.

    Args:
        data_root: Path to NuScenes data directory
        sam3_root: Path to SAM3 pre-generated masks
        output_dir: Path to output directory

    Returns:
        Dict with keys: data_root, sam3_root, output_dir,
                        nuscenes_info_train, nuscenes_info_val,
                        sam3_index_train, sam3_index_val
    """
    project_root = get_project_root()

    dr = Path(data_root or os.environ.get(
        'NUSCENES_ROOT', str(project_root / 'data' / 'nuscenes')
    ))
    if sam3_root:
        sr = Path(sam3_root)
    elif os.environ.get('SAM3_MASK_ROOT'):
        sr = Path(os.environ['SAM3_MASK_ROOT'])
    else:
        # Auto-detect: prefer included compressed data, fallback to raw
        compressed = Path(__file__).parent / 'data' / 'sam3_masks'
        raw = project_root / 'GEN_MASK_NUSCENCES_SAM'
        sr = compressed if compressed.exists() else raw
    od = Path(output_dir or os.environ.get(
        'OVSCAN_OUTPUT', str(project_root / 'Implement_OVSCAN' / 'output')
    ))

    return {
        'data_root': dr,
        'sam3_root': sr,
        'output_dir': od,
        'nuscenes_info_train': dr / 'nuscenes_infos_train.pkl',
        'nuscenes_info_val': dr / 'nuscenes_infos_val.pkl',
        'sam3_index_train': sr / 'train' / 'index.pkl',
        'sam3_index_val': sr / 'val' / 'index.pkl',
    }


# =============================================================================
# CAMERA NAMES
# =============================================================================

CAMERA_NAMES = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# =============================================================================
# NUSCENES CLASSES
# =============================================================================

NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(NUSCENES_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(NUSCENES_CLASSES)}

# =============================================================================
# NuScenes Attribute Mapping
# =============================================================================

DEFAULT_ATTRIBUTES = {
    'car': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'trailer': 'vehicle.parked',
    'bus': 'vehicle.parked',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'motorcycle': 'cycle.without_rider',
    'pedestrian': 'pedestrian.standing',
    'traffic_cone': '',
    'barrier': '',
}

# =============================================================================
# ANCHOR SIZES: [l, w, h] per class, [min, mean, max]
# =============================================================================

ANCHOR_SIZES = {
    'car': [
        [4.36, 1.87, 1.64],
        [4.63, 1.97, 1.74],
        [4.90, 2.07, 1.84],
    ],
    'truck': [
        [5.46, 2.25, 2.48],
        [6.93, 2.51, 2.84],
        [8.40, 2.78, 3.20],
    ],
    'construction_vehicle': [
        [5.02, 2.29, 2.77],
        [6.37, 2.85, 3.19],
        [7.72, 3.41, 3.61],
    ],
    'bus': [
        [9.68, 2.72, 3.18],
        [11.15, 2.93, 3.44],
        [12.62, 3.14, 3.70],
    ],
    'trailer': [
        [8.38, 2.60, 3.38],
        [10.24, 2.87, 3.87],
        [12.10, 3.14, 4.36],
    ],
    'barrier': [
        [0.40, 2.00, 0.80],
        [0.54, 2.59, 0.96],
        [0.70, 3.20, 1.15],
    ],
    'motorcycle': [
        [1.69, 0.65, 1.27],
        [1.95, 0.74, 1.41],
        [2.21, 0.83, 1.55],
    ],
    'bicycle': [
        [1.59, 0.51, 1.20],
        [1.76, 0.60, 1.44],
        [1.93, 0.69, 1.68],
    ],
    'pedestrian': [
        [0.60, 0.55, 1.54],
        [0.73, 0.67, 1.74],
        [0.86, 0.79, 1.94],
    ],
    'traffic_cone': [
        [0.35, 0.35, 0.95],
        [0.41, 0.41, 1.07],
        [0.47, 0.47, 1.19],
    ],
}

# =============================================================================
# 3D NMS PARAMETERS
# =============================================================================

NMS_IOU_THRESHOLD = 0.5

CLASS_NMS_IOU_THRESHOLDS = {
    'car': 0.5,
    'truck': 0.5,
    'bus': 0.5,
    'trailer': 0.5,
    'construction_vehicle': 0.5,
    'motorcycle': 0.35,
    'bicycle': 0.35,
    'pedestrian': 0.4,
    'traffic_cone': 0.3,
    'barrier': 0.4,
}

# =============================================================================
# DEPTH CLUSTERING PARAMETERS (DBSCAN)
# =============================================================================

DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 3

DEPTH_THRESHOLDS = {
    'car': 5.0,
    'truck': 8.0,
    'bus': 12.0,
    'trailer': 12.0,
    'construction_vehicle': 8.0,
    'motorcycle': 3.5,
    'bicycle': 3.5,
    'pedestrian': 4.0,
    'traffic_cone': 2.5,
    'barrier': 3.0,
}

ADAPTIVE_DBSCAN_EPS = {
    'car': 0.6,
    'truck': 1.2,
    'bus': 1.5,
    'trailer': 2.0,
    'construction_vehicle': 1.2,
    'motorcycle': 0.6,
    'bicycle': 0.5,
    'pedestrian': 0.45,
    'traffic_cone': 0.4,
    'barrier': 0.5,
}

# =============================================================================
# MINIMUM POINTS THRESHOLDS
# =============================================================================

MIN_POINTS_THRESHOLD = {
    'car': 5,
    'truck': 5,
    'bus': 5,
    'trailer': 5,
    'construction_vehicle': 5,
    'motorcycle': 3,
    'bicycle': 3,
    'pedestrian': 3,
    'traffic_cone': 2,
    'barrier': 3,
}

MIN_POINTS_FOR_OUTPUT = {
    'car': 5,
    'truck': 5,
    'bus': 5,
    'trailer': 5,
    'construction_vehicle': 5,
    'motorcycle': 4,
    'bicycle': 3,
    'pedestrian': 3,
    'traffic_cone': 2,
    'barrier': 3,
}

# =============================================================================
# DEPTH ESTIMATION SEARCH RADIUS (pixels)
# =============================================================================

DEPTH_SEARCH_RADIUS_PIXELS = {
    'car': 60,
    'truck': 80,
    'bus': 100,
    'trailer': 100,
    'construction_vehicle': 80,
    'motorcycle': 35,
    'bicycle': 30,
    'pedestrian': 40,
    'traffic_cone': 25,
    'barrier': 45,
}

# =============================================================================
# VOXEL PARAMETERS
# =============================================================================

VOXEL_SIZE = (0.2, 0.2, 0.2)
DEPTH_RANGE = (0.5, 80.0)

# =============================================================================
# CHECKPOINT PARAMETERS
# =============================================================================

CHECKPOINT_INTERVAL = 10

# =============================================================================
# SC-NOD PSO HYPERPARAMETERS (Paper Table 6)
# =============================================================================

N_SWARM = 50
N_ITER = 500
W_INIT = 10.0
W_END = 0.1
C1 = 1.0
C2 = 1.0
C_NOISE = 0.1

# =============================================================================
# COST FUNCTION WEIGHTS (Paper Eq. 2, 7)
# =============================================================================

LAMBDA1 = 5.0   # Density weight (Eq. 3-4)
LAMBDA2 = 1.0   # L-shape weight (Eq. 5)
LAMBDA3 = 1.0   # Surface weight (Eq. 6)
GAMMA = 3.0     # 2D IoU weight (Eq. 7)

# =============================================================================
# COST FUNCTION CONSTANTS
# =============================================================================

C_SURFACE = 5.0  # Surface distance clipping constant (meters)
