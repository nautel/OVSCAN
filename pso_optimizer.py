"""
Adaptive PSO Optimizer for SC-NOD 3D Box Search.

Follows paper Table 6:
  - Cosine annealing inertia: w(t) = w_end + 0.5*(w_init - w_end)*(1 + cos(pi*t/T))
  - Swarm size 50, cognitive/social c1=c2=1.0
  - Initialization: half at frustum ray center, half at point mean
  - Particle state: theta = (x, y, z, l, w, h, yaw)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from .cost_functions import (
    total_cost_batch, count_points_inside_box,
    project_3d_bbox_to_2d, compute_iou_2d,
)
from .config import (
    N_SWARM, N_ITER, W_INIT, W_END, C1, C2, C_NOISE,
    LAMBDA1, LAMBDA2, LAMBDA3, GAMMA, C_SURFACE,
    ANCHOR_SIZES,
)


class AdaptivePSOOptimizer:
    """
    PSO optimizer following SC-NOD paper formulation.

    Each particle encodes theta = (x, y, z, l, w, h, yaw).
    Minimizes the total cost J(theta).
    """

    def __init__(self, n_swarm=N_SWARM, n_iter=N_ITER,
                 w_init=W_INIT, w_end=W_END,
                 c1=C1, c2=C2, c_noise=C_NOISE):
        self.n_swarm = n_swarm
        self.n_iter = n_iter
        self.w_init = w_init
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.c_noise = c_noise

    def _cosine_annealing_inertia(self, t: int, T: int) -> float:
        """Cosine annealing: w(t) = w_end + 0.5*(w_init - w_end)*(1 + cos(pi*t/T))"""
        return self.w_end + 0.5 * (self.w_init - self.w_end) * (
            1.0 + np.cos(np.pi * t / max(T, 1))
        )

    def _initialize_particles(self, points: np.ndarray,
                              anchor_size: np.ndarray,
                              frustum_center: np.ndarray,
                              ground_z: float) -> np.ndarray:
        """
        Initialize particle positions.

        Half near the frustum ray center point, half near point cloud mean.
        """
        n = self.n_swarm
        particles = np.zeros((n, 7))

        pt_mean = points[:, :3].mean(axis=0)
        mean_dim = (anchor_size[0] + anchor_size[1]) / 2
        noise_std = self.c_noise * mean_dim

        half = n // 2
        particles[:half, 0] = frustum_center[0] + np.random.randn(half) * noise_std
        particles[:half, 1] = frustum_center[1] + np.random.randn(half) * noise_std
        particles[:half, 2] = ground_z + anchor_size[2] / 2 + np.random.randn(half) * 0.2

        rest = n - half
        particles[half:, 0] = pt_mean[0] + np.random.randn(rest) * noise_std
        particles[half:, 1] = pt_mean[1] + np.random.randn(rest) * noise_std
        particles[half:, 2] = ground_z + anchor_size[2] / 2 + np.random.randn(rest) * 0.2

        for dim in range(3):
            lo = 0.8 * anchor_size[dim]
            hi = 1.2 * anchor_size[dim]
            particles[:, 3 + dim] = np.random.uniform(lo, hi, n)

        particles[:, 6] = np.random.uniform(0, np.pi, n)

        return particles

    def optimize(self, points: np.ndarray, bbox_2d: np.ndarray,
                 ego_pos: np.ndarray, anchor_size: np.ndarray,
                 lidar2cam: np.ndarray, intrinsic: np.ndarray,
                 image_shape: Tuple[int, int],
                 frustum_center: np.ndarray,
                 ground_z: float,
                 n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        Run PSO optimization for a single object with given anchor.

        Args:
            points: (N, 3) object point cloud
            bbox_2d: (4,) [x1, y1, x2, y2] 2D detection box
            ego_pos: (3,) ego position in LiDAR frame
            anchor_size: (3,) [l, w, h] anchor to use
            lidar2cam: (4, 4) transformation
            intrinsic: (3, 3) camera intrinsic
            image_shape: (H, W)
            frustum_center: (3,) backprojected center point
            ground_z: estimated ground Z level
            n_iter: override number of iterations

        Returns:
            Dict with best_params, best_cost, etc.
        """
        max_iter = n_iter if n_iter is not None else self.n_iter
        n = self.n_swarm

        positions = self._initialize_particles(
            points, anchor_size, frustum_center, ground_z
        )

        pt_min = points[:, :3].min(axis=0)
        pt_max = points[:, :3].max(axis=0)
        margin = max(anchor_size[0], anchor_size[1])
        lb = np.array([
            pt_min[0] - margin, pt_min[1] - margin,
            ground_z - 0.5,
            anchor_size[0] * 0.5, anchor_size[1] * 0.5, anchor_size[2] * 0.5,
            0.0,
        ])
        ub = np.array([
            pt_max[0] + margin, pt_max[1] + margin,
            ground_z + anchor_size[2] * 2,
            anchor_size[0] * 1.5, anchor_size[1] * 1.5, anchor_size[2] * 1.5,
            np.pi,
        ])

        positions = np.clip(positions, lb, ub)

        v_max = (ub - lb) * 0.2
        velocities = np.random.uniform(-v_max, v_max, (n, 7))

        costs = total_cost_batch(
            positions, points, ego_pos, bbox_2d,
            lidar2cam, intrinsic, image_shape,
            LAMBDA1, LAMBDA2, LAMBDA3, GAMMA, C_SURFACE,
        )

        pbest_pos = positions.copy()
        pbest_cost = costs.copy()

        gbest_idx = np.argmin(costs)
        gbest_pos = positions[gbest_idx].copy()
        gbest_cost = costs[gbest_idx]

        for t in range(max_iter):
            w = self._cosine_annealing_inertia(t, max_iter)

            r1 = np.random.rand(n, 7)
            r2 = np.random.rand(n, 7)

            velocities = (
                w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )

            velocities = np.clip(velocities, -v_max, v_max)
            positions = positions + velocities
            positions = np.clip(positions, lb, ub)

            costs = total_cost_batch(
                positions, points, ego_pos, bbox_2d,
                lidar2cam, intrinsic, image_shape,
                LAMBDA1, LAMBDA2, LAMBDA3, GAMMA, C_SURFACE,
            )

            improved = costs < pbest_cost
            pbest_pos[improved] = positions[improved]
            pbest_cost[improved] = costs[improved]

            best_idx = np.argmin(pbest_cost)
            if pbest_cost[best_idx] < gbest_cost:
                gbest_cost = pbest_cost[best_idx]
                gbest_pos = pbest_pos[best_idx].copy()

        return {
            'best_params': gbest_pos,
            'best_cost': gbest_cost,
            'n_iter': max_iter,
            'n_swarm': n,
        }

    def optimize_multi_anchor(self, points: np.ndarray, bbox_2d: np.ndarray,
                              ego_pos: np.ndarray, class_name: str,
                              lidar2cam: np.ndarray, intrinsic: np.ndarray,
                              image_shape: Tuple[int, int],
                              frustum_center: np.ndarray,
                              ground_z: float,
                              n_iter: Optional[int] = None) -> Dict[str, Any]:
        """
        Try all 3 anchor sizes (min, mean, max) and pick the best.

        Returns:
            Best result dict with bbox_3d_center, bbox_3d_size, bbox_3d_yaw, score, etc.
        """
        anchors = ANCHOR_SIZES.get(class_name, ANCHOR_SIZES['car'])

        best_result = None
        best_cost = float('inf')

        for anchor_idx, anchor in enumerate(anchors):
            anchor_arr = np.array(anchor, dtype=np.float64)
            result = self.optimize(
                points=points,
                bbox_2d=bbox_2d,
                ego_pos=ego_pos,
                anchor_size=anchor_arr,
                lidar2cam=lidar2cam,
                intrinsic=intrinsic,
                image_shape=image_shape,
                frustum_center=frustum_center,
                ground_z=ground_z,
                n_iter=n_iter,
            )

            if result['best_cost'] < best_cost:
                best_cost = result['best_cost']
                best_result = result
                best_result['anchor_idx'] = anchor_idx

        params = best_result['best_params']
        center = params[:3]
        size = np.abs(params[3:6])
        size = np.maximum(size, 0.1)
        yaw = params[6]

        n_inside = count_points_inside_box(points, center, size, yaw)

        proj = project_3d_bbox_to_2d(center, size, yaw, lidar2cam, intrinsic, image_shape)
        iou_2d = 0.0
        if proj is not None:
            iou_2d = compute_iou_2d(proj, bbox_2d)

        density = n_inside / max(len(points), 1)
        score = 0.5 * density + 0.5 * iou_2d

        return {
            'bbox_3d_center': center,
            'bbox_3d_size': size,
            'bbox_3d_yaw': yaw,
            'score': float(score),
            'points_inside': int(n_inside),
            'iou_2d': float(iou_2d),
            'pso_cost': float(best_cost),
            'anchor_idx': best_result['anchor_idx'],
            'method': 'SC_NOD_PSO',
            'score_density': float(density),
            'score_surface': 0.0,
            'score_iou': float(iou_2d),
        }
