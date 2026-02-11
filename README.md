# OVSCAN: Open-Vocabulary 3D Object Detection

3D bounding box optimization for autonomous driving using LiDAR-camera fusion with open-vocabulary 2D detection.

## Pipeline Overview

```
Pre-generated SAM3 Masks + NuScenes LiDAR
    |
    v
Point-in-Mask Extraction (per camera)
    |
    v
DBSCAN Depth Clustering (noise removal)
    |
    v
3D BBox Optimization (PSO or Geometric)
    |
    v
3D NMS (multi-camera fusion)
    |
    v
NuScenes Submission JSON
```

## Installation

```bash
pip install numpy scipy scikit-learn tqdm

# Recommended (significantly improves accuracy and speed)
pip install numba shapely pyquaternion

# For NuScenes evaluation
pip install nuscenes-devkit
```

**Python**: >= 3.7 (tested with 3.8)

## Data Setup

### 1. NuScenes Dataset

Download [NuScenes v1.0-mini](https://www.nuscenes.org/nuscenes) and generate mmdetection3d info files:

```
data/nuscenes/
├── nuscenes_infos_train.pkl    # mmdetection3d format
├── nuscenes_infos_val.pkl
└── samples/
    └── LIDAR_TOP/
        └── *.pcd.bin
```

### 2. SAM3 Masks (included as compressed data)

Pre-generated SAM3 masks are included in `data/sam3_masks/` as compressed `.npz` files (~15MB total, 323 train + 1 val samples).

The data loader automatically supports both `.npy` (raw) and `.npz` (compressed) formats.

If you need to generate masks from scratch or decompress to raw `.npy`:

```bash
# Decompress .npz -> .npy (optional, not needed for running)
python -m Implement_OVSCAN.scripts.compress_masks \
    --decompress \
    --src_root Implement_OVSCAN/data/sam3_masks \
    --dst_root GEN_MASK_NUSCENCES_SAM

# Compress raw masks (if you generated new ones)
python -m Implement_OVSCAN.scripts.compress_masks \
    --src_root GEN_MASK_NUSCENCES_SAM \
    --dst_root Implement_OVSCAN/data/sam3_masks
```

### 3. Pre-computed Results (included)

Best submission and evaluation results are in `results/`:

```
results/
├── submissions/
│   └── scnod_pso_mAP24.40_NDS24.70.json   # Best submission (4.2MB)
└── evaluation/
    ├── BENCHMARK_SUMMARY.md                 # Ranked table of 99 submissions
    ├── all_results.json                     # Machine-readable results
    └── comparison_table.csv                 # CSV for analysis
```

To re-evaluate the included submission:

```bash
python -m Implement_OVSCAN.evaluate \
    --result_path Implement_OVSCAN/results/submissions/scnod_pso_mAP24.40_NDS24.70.json \
    --version v1.0-mini --eval_set mini_train --verbose
```

## Quick Start

```bash
# SC-NOD PSO optimizer (best accuracy, ~27s/sample)
python -m Implement_OVSCAN --split train --optimizer pso --verbose

# UltraFast geometric optimizer (~4s/sample)
python -m Implement_OVSCAN --split train --optimizer fast

# Process subset
python -m Implement_OVSCAN --split train --start_idx 0 --end_idx 10 --verbose

# Custom data paths
python -m Implement_OVSCAN --split train \
    --data_root /path/to/nuscenes \
    --sam3_root /path/to/sam3_masks \
    --output_dir /path/to/output

# Evaluate results
python -m Implement_OVSCAN.evaluate \
    --result_path output/submission_pso_*.json \
    --version v1.0-mini --eval_set mini_train
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NUSCENES_ROOT` | NuScenes data directory | `<project>/data/nuscenes` |
| `SAM3_MASK_ROOT` | SAM3 pre-generated masks | auto-detect `data/sam3_masks/` or `GEN_MASK_NUSCENCES_SAM/` |
| `OVSCAN_OUTPUT` | Output directory | `<project>/Implement_OVSCAN/output` |

## Optimizers

### SC-NOD PSO (default)
Particle Swarm Optimization following SC-NOD paper formulation:
- Cost: J = lambda1 * J_density + lambda2 * J_lshape + lambda3 * J_surface + gamma * J_2d
- Cosine annealing inertia, 50 particles, 500 iterations
- Multi-anchor: tries min/mean/max sizes per class
- Best mAP: **24.40%**, NDS: **24.70%**

### UltraFast Geometric
Geometric optimization with ConvexHull + grid search:
- ConvexHull yaw -> L-shape center correction -> grid search
- <100ms per object, class-specific strategies
- Good for rapid prototyping and large-scale processing

## Benchmark Results (NuScenes v1.0-mini, mini_train)

| Optimizer | mAP | NDS | car AP | ped AP | cone AP | mATE | Time/sample |
|-----------|-----|-----|--------|--------|---------|------|-------------|
| SC-NOD PSO | **24.40%** | **24.70%** | **28.1%** | 45.4% | 46.4% | **0.5614** | ~27s |
| UltraFast | 17.00% | 21.10% | 26.8% | 24.5% | 12.7% | 0.7133 | ~4s |

## Package Structure

```
Implement_OVSCAN/
├── __init__.py          # Package exports
├── __main__.py          # Entry point for python -m Implement_OVSCAN
├── config.py            # All configuration (paths, anchors, thresholds)
├── data_loader.py       # NuScenes + SAM3 mask loading (.npy and .npz)
├── mask_processor.py    # Object extraction from masks + DBSCAN
├── point_clustering.py  # DBSCAN depth clustering
├── cost_functions.py    # SC-NOD cost functions (J_density, J_lshape, J_surface, J_2d)
├── pso_optimizer.py     # AdaptivePSOOptimizer (SC-NOD paper)
├── fast_optimizer.py    # UltraFastOptimizer (geometric fallback)
├── nms_3d.py            # 3D NMS (Shapely BEV IoU)
├── output_formatter.py  # NuScenes submission JSON formatting
├── run.py               # Main CLI entry point
├── evaluate.py          # NuScenes evaluation wrapper
├── scripts/
│   └── compress_masks.py  # Compress/decompress SAM3 masks
├── data/
│   └── sam3_masks/        # Compressed SAM3 masks (~15MB)
├── results/
│   ├── submissions/       # Best submission JSON
│   └── evaluation/        # Benchmark summaries
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## License

This project is released under the [MIT License](LICENSE).
