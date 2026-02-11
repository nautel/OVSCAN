"""
NuScenes evaluation script for optimized 3D bounding box results.

Uses official NuScenes evaluation metrics:
- mAP (mean Average Precision)
- NDS (NuScenes Detection Score)
- Per-class AP

Usage:
    python -m Implement_OVSCAN.evaluate --result_path output/results_*.json --version v1.0-mini
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np

from .config import NUSCENES_CLASSES, get_paths


def run_nuscenes_eval(
    result_path: str,
    version: str = 'v1.0-mini',
    data_root: Optional[str] = None,
    eval_set: str = 'mini_train',
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run official NuScenes evaluation.

    Args:
        result_path: Path to results JSON file
        version: NuScenes version ('v1.0-mini', 'v1.0-trainval')
        data_root: Path to NuScenes data
        eval_set: Evaluation set ('mini_train', 'mini_val', 'train', 'val')
        output_dir: Directory to save evaluation results
        verbose: Print detailed output

    Returns:
        Dict with evaluation metrics
    """
    paths = get_paths(data_root=data_root, output_dir=output_dir)
    dr = str(paths['data_root'])
    od = str(paths['output_dir'])

    # Add nuscenes-devkit to path if available
    project_root = Path(__file__).parent.parent
    devkit_path = project_root / 'nuscenes-devkit' / 'python-sdk'
    if devkit_path.exists():
        sys.path.insert(0, str(devkit_path))

    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        print("Error: nuscenes-devkit not installed or not in path.")
        print("Install with: pip install nuscenes-devkit")
        return {}

    print("=" * 80)
    print("NuScenes Official Evaluation")
    print("=" * 80)
    print(f"\nResult file: {result_path}")
    print(f"Version: {version}")
    print(f"Eval set: {eval_set}")
    print(f"Data root: {dr}")

    print("\nLoading NuScenes...")
    nusc = NuScenes(version=version, dataroot=dr, verbose=verbose)

    cfg = config_factory('detection_cvpr_2019')

    output_path = Path(od) / 'eval_results'
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nRunning evaluation...")
    nusc_eval = NuScenesEval(
        nusc=nusc,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=str(output_path),
        verbose=verbose,
    )

    metrics = nusc_eval.main(plot_examples=0, render_curves=False)

    results = {
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
        'mATE': metrics['tp_errors']['trans_err'] if 'tp_errors' in metrics else None,
        'mASE': metrics['tp_errors']['scale_err'] if 'tp_errors' in metrics else None,
        'mAOE': metrics['tp_errors']['orient_err'] if 'tp_errors' in metrics else None,
        'mAVE': metrics['tp_errors']['vel_err'] if 'tp_errors' in metrics else None,
        'mAAE': metrics['tp_errors']['attr_err'] if 'tp_errors' in metrics else None,
    }

    if 'label_aps' in metrics:
        results['per_class_ap'] = {}
        for class_name, ap_dict in metrics['label_aps'].items():
            results['per_class_ap'][class_name] = np.mean(list(ap_dict.values()))

    if 'label_tp_errors' in metrics:
        results['per_class_errors'] = {}
        for class_name, err_dict in metrics['label_tp_errors'].items():
            results['per_class_errors'][class_name] = {
                'ATE': err_dict.get('trans_err', 1.0),
                'ASE': err_dict.get('scale_err', 1.0),
                'AOE': err_dict.get('orient_err', 1.0),
                'AVE': err_dict.get('vel_err', 1.0),
                'AAE': err_dict.get('attr_err', 1.0),
            }

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nmAP: {results['mAP']:.4f}")
    print(f"NDS: {results['NDS']:.4f}")

    if results.get('mATE'):
        print(f"\nmATE: {results['mATE']:.4f}")
        print(f"mASE: {results['mASE']:.4f}")
        print(f"mAOE: {results['mAOE']:.4f}")
        print(f"mAVE: {results['mAVE']:.4f}")
        print(f"mAAE: {results['mAAE']:.4f}")

    if results.get('per_class_ap'):
        print("\nPer-class AP:")
        for class_name in NUSCENES_CLASSES:
            if class_name in results['per_class_ap']:
                ap = results['per_class_ap'][class_name]
                print(f"  {class_name:25s}: {ap:.4f}")

    if results.get('per_class_errors'):
        print("\nPer-class TP Errors (ATE / ASE / AOE):")
        for class_name in NUSCENES_CLASSES:
            if class_name in results['per_class_errors']:
                e = results['per_class_errors'][class_name]
                print(f"  {class_name:25s}: ATE={e['ATE']:.4f}  ASE={e['ASE']:.4f}  AOE={e['AOE']:.4f}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f'metrics_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nMetrics saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate 3D BBox optimization results using NuScenes metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate JSON results
    python -m Implement_OVSCAN.evaluate --result_path output/results.json --version v1.0-mini

    # Evaluate with custom data root
    python -m Implement_OVSCAN.evaluate --result_path output/results.json --data_root /path/to/nuscenes
        """,
    )

    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval'],
                        help='NuScenes dataset version')
    parser.add_argument('--eval_set', type=str, default='mini_train',
                        help='Evaluation set (mini_train, mini_val, train, val)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='NuScenes data root')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for eval results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    run_nuscenes_eval(
        result_path=args.result_path,
        version=args.version,
        data_root=args.data_root,
        eval_set=args.eval_set,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
