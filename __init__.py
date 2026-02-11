"""
OVSCAN: Open-Vocabulary 3D Object Detection via LiDAR-Camera Fusion.

This package provides 3D bounding box optimization for autonomous driving,
combining 2D open-vocabulary detection (SAM3 masks) with LiDAR point cloud
processing.

Two optimizers are available:
- SC-NOD PSO (AdaptivePSOOptimizer): Best accuracy, ~27s/sample
- UltraFast Geometric (UltraFastOptimizer): Fast fallback, <100ms/object
"""

__version__ = '1.0.0'

from .config import (
    NUSCENES_CLASSES,
    ANCHOR_SIZES,
    get_paths,
)
from .data_loader import MaskNuScenesLoader, load_sample_data
from .mask_processor import MaskProcessor, process_all_cameras
from .pso_optimizer import AdaptivePSOOptimizer
from .fast_optimizer import UltraFastOptimizer
from .nms_3d import nms_3d_bboxes
from .output_formatter import NuScenesFormatter, format_results_for_nuscenes

__all__ = [
    'NUSCENES_CLASSES',
    'ANCHOR_SIZES',
    'get_paths',
    'MaskNuScenesLoader',
    'load_sample_data',
    'MaskProcessor',
    'process_all_cameras',
    'AdaptivePSOOptimizer',
    'UltraFastOptimizer',
    'nms_3d_bboxes',
    'NuScenesFormatter',
    'format_results_for_nuscenes',
]
