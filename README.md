# OV-SCAN: Open-Vocabulary 3D Object Detection

Non-official implementation of the SC-NOD 3D bounding box optimization from [OV-SCAN (ICCV 2025)](https://arxiv.org/abs/2503.06435).

> **OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection**
> Adrian Chow et al. | [Paper](https://arxiv.org/abs/2503.06435)

```
SAM3 Masks + NuScenes LiDAR → Point-in-Mask → DBSCAN → 3D BBox Optimization → 3D NMS → Submission JSON
```

## Results (NuScenes v1.0-mini)

| Optimizer | mAP | NDS | car | ped | cone | mATE | Speed |
|-----------|-----|-----|-----|-----|------|------|-------|
| **SC-NOD PSO** | **24.40%** | **24.70%** | 28.1% | 45.4% | 46.4% | 0.561 | ~27s/sample |
| UltraFast | 17.00% | 21.10% | 26.8% | 24.5% | 12.7% | 0.713 | ~4s/sample |

Pre-computed submission included in `results/submissions/`.

## Setup

```bash
# 1. Clone
git clone https://github.com/nautel/OVSCAN.git && cd OVSCAN

# 2. Install dependencies
pip install numpy scipy scikit-learn tqdm numba shapely pyquaternion

# 3. Download NuScenes v1.0-mini (LiDAR only)
#    From: https://www.nuscenes.org/nuscenes
#    Extract so that data/nuscenes/samples/LIDAR_TOP/*.bin exists
#    Info PKL files and SAM3 masks are already included in this repo.

# 4. (Optional) For evaluation
pip install nuscenes-devkit
```

**What's included in this repo (no extra downloads needed besides NuScenes raw data):**
- `data/nuscenes/*.pkl` — mmdetection3d info files (train + val)
- `data/sam3_masks/` — Compressed SAM3 masks for all 323 train + 1 val samples (26MB, 500x compressed)

## Quick Start

```bash
# SC-NOD PSO optimizer (best accuracy)
python -m Implement_OVSCAN --split train --optimizer pso --verbose

# UltraFast geometric optimizer (fast)
python -m Implement_OVSCAN --split train --optimizer fast

# Process subset
python -m Implement_OVSCAN --split train --start_idx 0 --end_idx 10 --verbose

# Evaluate
python -m Implement_OVSCAN.evaluate \
    --result_path results/submissions/scnod_pso_mAP24.40_NDS24.70.json \
    --version v1.0-mini --eval_set mini_train --verbose
```

Custom paths: `--data_root /path/to/nuscenes --sam3_root /path/to/masks --output_dir /path/to/output`

## Package Structure

```
├── Implement_OVSCAN/
│   ├── config.py            # Paths, anchors, thresholds, PSO hyperparameters
│   ├── cost_functions.py    # SC-NOD cost: J_density, J_lshape, J_surface, J_2d
│   ├── pso_optimizer.py     # AdaptivePSOOptimizer (cosine annealing, multi-anchor)
│   ├── fast_optimizer.py    # UltraFastOptimizer (ConvexHull + grid search)
│   ├── data_loader.py       # NuScenes + SAM3 mask loading (.npy/.npz)
│   ├── mask_processor.py    # LiDAR-to-camera projection + object extraction
│   ├── point_clustering.py  # DBSCAN depth clustering
│   ├── nms_3d.py            # 3D NMS (Shapely BEV IoU)
│   ├── output_formatter.py  # NuScenes submission JSON
│   ├── run.py               # CLI entry point
│   └── evaluate.py          # NuScenes evaluation wrapper
├── data/
│   ├── nuscenes/            # Info PKLs (included) + LiDAR bins (download)
│   └── sam3_masks/          # Compressed masks (included, 26MB)
└── results/                 # Pre-computed submission + benchmark
```

## Citation

```bibtex
@inproceedings{chow2025ovscan,
    title={OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection},
    author={Chow, Adrian},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## License

[MIT License](LICENSE)
