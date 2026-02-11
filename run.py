"""
Main CLI entry point for OVSCAN 3D BBox Optimization.

Pipeline:
  1. Load NuScenes data + SAM3 masks
  2. Extract objects via MaskProcessor (DBSCAN depth clustering)
  3. For each object: run optimizer (PSO or UltraFast geometric)
  4. Apply 3D NMS
  5. Format and save NuScenes submission JSON

Usage:
  python -m Implement_OVSCAN.run --split train --start_idx 0 --end_idx 10 --verbose
  python -m Implement_OVSCAN.run --split train --optimizer fast
  python -m Implement_OVSCAN.run --split train --optimizer pso --n_iter 1000
"""

import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

from .data_loader import MaskNuScenesLoader, load_sample_data
from .mask_processor import MaskProcessor, process_all_cameras
from .nms_3d import nms_3d_bboxes
from .output_formatter import NuScenesFormatter, format_results_for_nuscenes
from .fast_optimizer import (
    estimate_depth_for_small_object,
    create_bbox_from_depth_estimate,
    estimate_ground_z_from_points,
    UltraFastOptimizer,
)
from .pso_optimizer import AdaptivePSOOptimizer
from .config import (
    NMS_IOU_THRESHOLD, CHECKPOINT_INTERVAL,
    MIN_POINTS_FOR_OUTPUT, ANCHOR_SIZES, N_ITER,
    DEPTH_SEARCH_RADIUS_PIXELS,
    LAMBDA1, LAMBDA2, LAMBDA3, GAMMA,
    get_paths,
)


def backproject_bbox_center_to_lidar(bbox_2d: np.ndarray, depth: float,
                                     lidar2cam: np.ndarray,
                                     intrinsic: np.ndarray) -> np.ndarray:
    """Backproject 2D bbox center to 3D LiDAR frame at given depth."""
    cx = (bbox_2d[0] + bbox_2d[2]) / 2
    cy = (bbox_2d[1] + bbox_2d[3]) / 2

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx_int, cy_int = intrinsic[0, 2], intrinsic[1, 2]

    x_cam = (cx - cx_int) * depth / fx
    y_cam = (cy - cy_int) * depth / fy
    z_cam = depth

    cam2lidar = np.linalg.inv(lidar2cam)
    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    point_lidar = cam2lidar @ point_cam

    return point_lidar[:3]


def optimize_single_object_pso(obj: Dict[str, Any],
                               lidar_points: np.ndarray,
                               optimizer: AdaptivePSOOptimizer,
                               n_iter: Optional[int] = None,
                               verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Optimize a single object using SC-NOD PSO."""
    class_name = obj['class_name']
    bbox_2d = obj['bbox_2d']
    lidar2cam = obj['lidar2cam']
    intrinsic = obj['intrinsic']
    image_shape = obj['image_shape']
    score_2d = obj.get('score', 0.5)

    points = obj.get('filtered_points', obj['points'])
    ego_pos = np.array([0.0, 0.0, 0.0])
    min_pts = MIN_POINTS_FOR_OUTPUT.get(class_name, 3)

    if len(points) < min_pts:
        search_radius = DEPTH_SEARCH_RADIUS_PIXELS.get(class_name, 50)
        est_depth = estimate_depth_for_small_object(
            bbox_2d, lidar_points, lidar2cam, intrinsic, image_shape,
            search_radius_pixels=search_radius, verbose=verbose,
        )
        if est_depth is not None:
            result = create_bbox_from_depth_estimate(
                bbox_2d, est_depth, class_name, intrinsic, lidar2cam,
                detection_score=score_2d, verbose=verbose,
                lidar_points=lidar_points,
            )
            result['method'] = 'SC_NOD_DEPTH_FALLBACK'
            return result
        return None

    # Compute frustum center
    pts_homo = np.hstack([points[:, :3], np.ones((len(points), 1))])
    pts_cam = (lidar2cam @ pts_homo.T).T
    valid_depth = pts_cam[:, 2] > 0.1
    mean_depth_cam = np.median(pts_cam[valid_depth, 2]) if valid_depth.any() else None

    if mean_depth_cam is None or mean_depth_cam <= 0:
        mean_depth_cam = np.sqrt(np.sum(points[:, :3].mean(axis=0) ** 2))

    frustum_center = backproject_bbox_center_to_lidar(
        bbox_2d, mean_depth_cam, lidar2cam, intrinsic
    )

    pt_mean = points[:, :3].mean(axis=0)
    ground_z = estimate_ground_z_from_points(
        lidar_points, pt_mean[0], pt_mean[1],
        search_radius=5.0, default_ground_z=-1.7,
    )

    if verbose:
        print(f"    PSO: {class_name}, {len(points)} pts, "
              f"depth={mean_depth_cam:.1f}m, ground_z={ground_z:.2f}")

    result = optimizer.optimize_multi_anchor(
        points=points,
        bbox_2d=bbox_2d,
        ego_pos=ego_pos,
        class_name=class_name,
        lidar2cam=lidar2cam,
        intrinsic=intrinsic,
        image_shape=image_shape,
        frustum_center=frustum_center,
        ground_z=ground_z,
        n_iter=n_iter,
    )

    result['class_name'] = class_name
    result['detection_score'] = score_2d
    result['bbox_2d'] = bbox_2d
    result['score'] = result['score'] * score_2d

    if result['points_inside'] < min_pts:
        if verbose:
            print(f"    Filtered: only {result['points_inside']} pts inside (need {min_pts})")
        return None

    return result


def optimize_single_object_fast(obj: Dict[str, Any],
                                lidar_points: np.ndarray,
                                optimizer: UltraFastOptimizer,
                                verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Optimize a single object using UltraFast geometric method."""
    class_name = obj['class_name']
    bbox_2d = obj['bbox_2d']
    lidar2cam = obj['lidar2cam']
    intrinsic = obj['intrinsic']
    image_shape = obj['image_shape']
    score_2d = obj.get('score', 0.5)

    points = obj.get('filtered_points', obj['points'])
    min_pts = MIN_POINTS_FOR_OUTPUT.get(class_name, 3)

    if len(points) < min_pts:
        search_radius = DEPTH_SEARCH_RADIUS_PIXELS.get(class_name, 50)
        est_depth = estimate_depth_for_small_object(
            bbox_2d, lidar_points, lidar2cam, intrinsic, image_shape,
            search_radius_pixels=search_radius, verbose=verbose,
        )
        if est_depth is not None:
            result = create_bbox_from_depth_estimate(
                bbox_2d, est_depth, class_name, intrinsic, lidar2cam,
                detection_score=score_2d, verbose=verbose,
                lidar_points=lidar_points,
            )
            result['method'] = 'FAST_DEPTH_FALLBACK'
            return result
        return None

    anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['car'])
    anchor_size = np.array(anchors[1])

    result = optimizer.optimize(
        voxel_centers=points,
        points=points,
        bbox_2d_target=bbox_2d,
        anchor_size=anchor_size,
        lidar2cam=lidar2cam,
        intrinsic=intrinsic,
        image_shape=image_shape,
        class_name=class_name,
        extended_voxels=points,
    )

    result['class_name'] = class_name
    result['detection_score'] = score_2d
    result['bbox_2d'] = bbox_2d
    result['score'] = result['score'] * score_2d

    if result['points_inside'] < min_pts:
        if verbose:
            print(f"    Filtered: only {result['points_inside']} pts inside (need {min_pts})")
        return None

    return result


def process_single_sample(loader, processor, optimizer, sample_idx,
                          optimizer_type='pso', apply_nms=True,
                          n_iter=None, verbose=False):
    """Process a single sample through the full pipeline."""
    start_time = time.time()

    sample_data = load_sample_data(loader, sample_idx)
    sample_token = sample_data['sample_token']
    lidar_points = sample_data['lidar_points']

    if verbose:
        print(f"\n[{sample_idx}] Sample: {sample_token}")
        print(f"  LiDAR points: {len(lidar_points)}")

    all_objects = process_all_cameras(
        processor=processor,
        lidar_points=lidar_points,
        cameras_data=sample_data['cameras'],
        verbose=verbose,
    )

    if verbose:
        print(f"  Objects from masks: {len(all_objects)}")

    if len(all_objects) == 0:
        return {
            'sample_token': sample_token,
            'sample_idx': sample_idx,
            'results': [],
            'n_objects_input': 0,
            'n_objects_output': 0,
            'processing_time': time.time() - start_time,
        }

    results = []
    for obj in all_objects:
        if optimizer_type == 'pso':
            result = optimize_single_object_pso(
                obj, lidar_points, optimizer,
                n_iter=n_iter, verbose=verbose,
            )
        else:
            result = optimize_single_object_fast(
                obj, lidar_points, optimizer, verbose=verbose,
            )
        if result is not None:
            results.append(result)

    if verbose:
        print(f"  Optimized: {len(results)} objects")

    if apply_nms and len(results) > 1:
        n_before = len(results)
        results = nms_3d_bboxes(results, iou_threshold=NMS_IOU_THRESHOLD)
        if verbose:
            print(f"  After NMS: {len(results)} (removed {n_before - len(results)})")

    processing_time = time.time() - start_time

    return {
        'sample_token': sample_token,
        'sample_idx': sample_idx,
        'results': results,
        'n_objects_input': len(all_objects),
        'n_objects_output': len(results),
        'processing_time': processing_time,
    }


def save_checkpoint(all_results, processed_indices, checkpoint_path, metadata=None):
    """Save checkpoint for resume capability."""
    checkpoint = {
        'results': all_results,
        'processed_indices': processed_indices,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {len(processed_indices)} samples processed")


def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists."""
    if not checkpoint_path.exists():
        return None
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Loaded checkpoint: {len(checkpoint['processed_indices'])} samples already processed")
    return checkpoint


def run_batch_optimization(
    split='train',
    start_idx=0,
    end_idx=None,
    resume=False,
    apply_nms=True,
    verbose=False,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    n_iter=N_ITER,
    optimizer_type='pso',
    data_root=None,
    sam3_root=None,
    output_dir=None,
):
    """Run batch optimization on all samples."""
    print("=" * 80)
    print(f"OVSCAN 3D Box Optimization ({optimizer_type.upper()})")
    print("=" * 80)

    paths = get_paths(data_root=data_root, sam3_root=sam3_root, output_dir=output_dir)
    out_dir = paths['output_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitializing...")
    loader = MaskNuScenesLoader(split=split, data_root=data_root, sam3_root=sam3_root)
    processor = MaskProcessor(use_depth_clustering=True)

    if optimizer_type == 'pso':
        optimizer = AdaptivePSOOptimizer(n_iter=n_iter)
        print(f"  Optimizer: SC-NOD PSO (swarm={optimizer.n_swarm}, iter={n_iter})")
        print(f"  Weights: l1={LAMBDA1}, l2={LAMBDA2}, l3={LAMBDA3}, g={GAMMA}")
    else:
        optimizer = UltraFastOptimizer(try_all_anchors=True, verbose=verbose)
        print(f"  Optimizer: UltraFast Geometric")

    total_samples = len(loader)
    if end_idx is None:
        end_idx = total_samples
    end_idx = min(end_idx, total_samples)

    print(f"  Split: {split}")
    print(f"  Total available: {total_samples}")
    print(f"  Processing range: [{start_idx}, {end_idx})")

    # Checkpoint
    checkpoint_path = out_dir / f'checkpoint_{optimizer_type}_{split}_{start_idx}_{end_idx}.pkl'

    all_results = {}
    processed_indices = set()
    sample_transforms = {}

    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            all_results = checkpoint['results']
            processed_indices = set(checkpoint['processed_indices'])
            for sample_token in all_results.keys():
                try:
                    for idx in range(len(loader)):
                        if loader.get_sample_token(idx) == sample_token:
                            lidar2ego, ego2global = loader.get_transforms(sample_token)
                            sample_transforms[sample_token] = (lidar2ego, ego2global)
                            break
                except Exception:
                    pass
            print(f"  Resuming from {len(processed_indices)} processed samples")

    formatter = NuScenesFormatter(output_dir=out_dir)

    print(f"\nProcessing samples...")
    stats = {
        'total_objects_input': 0,
        'total_objects_output': 0,
        'total_time': 0,
        'failed_samples': [],
    }

    indices_to_process = [i for i in range(start_idx, end_idx) if i not in processed_indices]
    pbar = tqdm(indices_to_process, desc=f"OVSCAN ({optimizer_type})", unit="sample")

    for sample_idx in pbar:
        try:
            result = process_single_sample(
                loader=loader,
                processor=processor,
                optimizer=optimizer,
                sample_idx=sample_idx,
                optimizer_type=optimizer_type,
                apply_nms=apply_nms,
                n_iter=n_iter,
                verbose=verbose,
            )

            sample_token = result['sample_token']
            all_results[sample_token] = result['results']
            processed_indices.add(sample_idx)

            try:
                lidar2ego, ego2global = loader.get_transforms(sample_token)
                sample_transforms[sample_token] = (lidar2ego, ego2global)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not get transforms: {e}")

            stats['total_objects_input'] += result['n_objects_input']
            stats['total_objects_output'] += result['n_objects_output']
            stats['total_time'] += result['processing_time']

            pbar.set_postfix({
                'in': result['n_objects_input'],
                'out': result['n_objects_output'],
                't': f"{result['processing_time']:.1f}s",
            })

            if len(processed_indices) % checkpoint_interval == 0:
                save_checkpoint(
                    all_results, list(processed_indices),
                    checkpoint_path, {'stats': stats},
                )

        except Exception as e:
            print(f"\nError processing sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            stats['failed_samples'].append(sample_idx)
            continue

    save_checkpoint(all_results, list(processed_indices), checkpoint_path, {'stats': stats})

    print("\nSaving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_name = f'submission_{optimizer_type}_{split}_{start_idx}_{end_idx}_{timestamp}'

    print(f"  Transforms collected for {len(sample_transforms)} samples")

    output_paths = format_results_for_nuscenes(
        all_results,
        output_dir=out_dir,
        output_name=output_name,
        save_json=True,
        save_pkl=True,
        sample_transforms=sample_transforms,
    )

    # Summary
    print("\n" + "=" * 80)
    print("BATCH OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Samples processed: {len(processed_indices)}")
    print(f"  Failed samples: {len(stats['failed_samples'])}")
    print(f"  Total objects (input): {stats['total_objects_input']}")
    print(f"  Total objects (after NMS): {stats['total_objects_output']}")
    print(f"  Total time: {stats['total_time']:.1f}s")
    print(f"  Average time per sample: {stats['total_time'] / max(1, len(processed_indices)):.2f}s")

    print(f"\nOutput files:")
    for fmt, path in output_paths.items():
        print(f"  {fmt.upper()}: {path}")

    if stats['failed_samples']:
        print(f"\nFailed sample indices: {stats['failed_samples']}")

    return all_results, output_paths


def main():
    parser = argparse.ArgumentParser(
        description='OVSCAN 3D BBox Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # SC-NOD PSO (best accuracy, slow)
    python -m Implement_OVSCAN.run --split train --optimizer pso --start_idx 0 --end_idx 10 --verbose

    # UltraFast geometric (fast fallback)
    python -m Implement_OVSCAN.run --split train --optimizer fast

    # Full mini_train with PSO
    python -m Implement_OVSCAN.run --split train --optimizer pso

    # Resume from checkpoint
    python -m Implement_OVSCAN.run --split train --resume

    # Custom paths
    python -m Implement_OVSCAN.run --split train --data_root /path/to/nuscenes --sam3_root /path/to/sam3
        """,
    )

    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-nms', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument('--n_iter', type=int, default=N_ITER,
                        help=f'PSO iterations (default: {N_ITER})')
    parser.add_argument('--optimizer', type=str, default='pso',
                        choices=['pso', 'fast'],
                        help='Optimizer: pso (SC-NOD, best accuracy) or fast (geometric)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to NuScenes data directory')
    parser.add_argument('--sam3_root', type=str, default=None,
                        help='Path to SAM3 mask directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory')

    args = parser.parse_args()

    run_batch_optimization(
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resume=args.resume,
        apply_nms=not args.no_nms,
        verbose=args.verbose,
        checkpoint_interval=args.checkpoint_interval,
        n_iter=args.n_iter,
        optimizer_type=args.optimizer,
        data_root=args.data_root,
        sam3_root=args.sam3_root,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
