"""
Depth-aware point clustering for mask points.
Filters background noise based on object-specific depth thresholds.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional, Any

from .config import DEPTH_THRESHOLDS, ADAPTIVE_DBSCAN_EPS


def compute_depth_range(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute depth statistics of points.

    Args:
        points: (N, 3+) array of 3D points [x, y, z, ...]

    Returns:
        min_depth, max_depth, depth_span
    """
    if len(points) == 0:
        return 0.0, 0.0, 0.0

    depths = np.linalg.norm(points[:, :3], axis=1)
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    depth_span = max_depth - min_depth

    return min_depth, max_depth, depth_span


def filter_points_by_depth_clustering(
    points: np.ndarray,
    object_class: str,
    depth_threshold: Optional[float] = None,
    eps: float = 0.5,
    min_samples: int = 3,
    min_cluster_size: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Filter mask points using depth-aware clustering.

    Strategy:
    1. Compute depth span of all points
    2. If depth span exceeds threshold -> likely has background noise
    3. Apply DBSCAN clustering to separate foreground from background
    4. Keep the largest cluster (assumed to be the object)

    Args:
        points: (N, 3+) array of 3D points
        object_class: class name (e.g., 'car', 'person')
        depth_threshold: depth span threshold. If None, use class-specific
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        min_cluster_size: minimum cluster size to keep
        verbose: print debug info

    Returns:
        filtered_points: (M, 3+) filtered points
    """
    if len(points) == 0:
        return points

    if depth_threshold is None:
        depth_threshold = DEPTH_THRESHOLDS.get(object_class, 3.0)

    min_depth, max_depth, depth_span = compute_depth_range(points)

    if verbose:
        print(f"  Depth range: [{min_depth:.2f}, {max_depth:.2f}] m, span: {depth_span:.2f} m")
        print(f"  Threshold for '{object_class}': {depth_threshold:.2f} m")

    if depth_span <= depth_threshold:
        if verbose:
            print(f"  Depth span OK, keeping all {len(points)} points")
        return points

    if verbose:
        print(f"  Depth span exceeds threshold, applying clustering...")

    actual_eps = ADAPTIVE_DBSCAN_EPS.get(object_class, eps)
    if verbose and actual_eps != eps:
        print(f"  Using class-specific eps={actual_eps:.2f} for {object_class}")

    clustering = DBSCAN(eps=actual_eps, min_samples=min_samples).fit(points[:, :3])
    labels = clustering.labels_

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) == 0:
        if verbose:
            print(f"  All points classified as noise, filtering by proximity...")

        depths = np.linalg.norm(points[:, :3], axis=1)
        sorted_indices = np.argsort(depths)

        filtered_points = []
        first_point = points[sorted_indices[0]]
        filtered_points.append(first_point)

        for idx in sorted_indices[1:]:
            point = points[idx]
            distance_from_first = np.linalg.norm(point[:3] - first_point[:3])
            if distance_from_first <= depth_threshold:
                filtered_points.append(point)

        if len(filtered_points) < min_cluster_size and len(points) >= min_cluster_size:
            filtered_points = points[sorted_indices[:min_cluster_size]].tolist()
        elif len(filtered_points) == 0:
            filtered_points = points.tolist()

        filtered_points = np.array(filtered_points)

        if verbose:
            _, _, final_span = compute_depth_range(filtered_points)
            print(f"  Kept {len(filtered_points)} points within {depth_threshold:.2f}m of closest point")

        return filtered_points

    cluster_sizes = {}
    for label in unique_labels:
        cluster_sizes[label] = np.sum(labels == label)

    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    largest_cluster_size = cluster_sizes[largest_cluster_label]

    if verbose:
        print(f"  Found {len(unique_labels)} clusters")
        print(f"  Largest cluster: {largest_cluster_size} points (label={largest_cluster_label})")

    if largest_cluster_size < min_cluster_size:
        if verbose:
            print(f"  Largest cluster too small, keeping all points")
        return points

    filtered_points = points[labels == largest_cluster_label]

    if verbose:
        _, _, filtered_depth_span = compute_depth_range(filtered_points)
        print(f"  Filtered: {len(points)} -> {len(filtered_points)} points")
        print(f"  Filtered depth span: {filtered_depth_span:.2f} m")

    return filtered_points


def filter_multiple_objects_by_depth(
    all_objects: List[Dict[str, Any]],
    depth_thresholds: Optional[Dict[str, float]] = None,
    eps: float = 0.5,
    min_samples: int = 3,
    min_cluster_size: int = 3,
    verbose: bool = False,
    use_adaptive_eps: bool = False,
) -> List[Dict[str, Any]]:
    """
    Apply depth-aware filtering to multiple objects.

    Keeps original 'points' unchanged, adds 'filtered_points' key.

    Args:
        all_objects: list of object dicts with 'points' and 'label_text'
        depth_thresholds: optional custom thresholds per class
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter
        min_cluster_size: minimum cluster size
        verbose: print debug info
        use_adaptive_eps: if True, auto-estimate eps per object

    Returns:
        all_objects with added 'filtered_points', 'original_point_count',
        'filtered_point_count', 'depth_filter_applied'
    """
    if depth_thresholds is None:
        depth_thresholds = DEPTH_THRESHOLDS

    for idx, obj in enumerate(all_objects):
        points = obj.get('points', np.array([]).reshape(0, 3))
        label_text = obj.get('label_text', 'unknown')

        obj['depth_filter_applied'] = False
        obj['original_point_count'] = len(points)

        if len(points) == 0:
            obj['filtered_points'] = points
            obj['filtered_point_count'] = 0
            continue

        if verbose:
            print(f"\nObject {idx} ({label_text}):")
            print(f"  Original points: {len(points)}")

        threshold = depth_thresholds.get(label_text, 3.0)

        if use_adaptive_eps:
            filtered_points = filter_with_adaptive_eps(
                points=points,
                object_class=label_text,
                depth_threshold=threshold,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                verbose=verbose,
            )
        else:
            filtered_points = filter_points_by_depth_clustering(
                points=points,
                object_class=label_text,
                depth_threshold=threshold,
                eps=eps,
                min_samples=min_samples,
                min_cluster_size=min_cluster_size,
                verbose=verbose,
            )

        obj['filtered_points'] = filtered_points
        obj['filtered_point_count'] = len(filtered_points)
        obj['depth_filter_applied'] = len(filtered_points) != len(points)

    return all_objects


def adaptive_eps_estimation(points: np.ndarray, k: int = 4) -> float:
    """
    Estimate optimal eps for DBSCAN based on k-nearest neighbor distance.

    Args:
        points: (N, 3) array of 3D points
        k: number of nearest neighbors

    Returns:
        eps: estimated eps value
    """
    if len(points) < k + 1:
        return 0.5

    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points[:, :3])
    distances, _ = nbrs.kneighbors(points[:, :3])

    k_distances = distances[:, k]
    eps = np.percentile(k_distances, 90)

    return float(eps)


def filter_with_adaptive_eps(
    points: np.ndarray,
    object_class: str,
    depth_threshold: Optional[float] = None,
    min_samples: int = 3,
    min_cluster_size: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Filter points with automatically estimated eps.

    Args:
        points: (N, 3+) array
        object_class: class name
        depth_threshold: depth span threshold
        min_samples: DBSCAN min_samples
        min_cluster_size: minimum cluster size
        verbose: print debug info

    Returns:
        filtered_points: (M, 3+) filtered points
    """
    if len(points) < 10:
        return points

    eps = adaptive_eps_estimation(points, k=min_samples)

    if verbose:
        print(f"  Estimated eps: {eps:.3f}")

    return filter_points_by_depth_clustering(
        points=points,
        object_class=object_class,
        depth_threshold=depth_threshold,
        eps=eps,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        verbose=verbose,
    )
