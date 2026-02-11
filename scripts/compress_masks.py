#!/usr/bin/env python3
"""
Compress SAM3 masks from .npy to .npz format.

SAM3 masks are extremely sparse (92% zeros), compressing ~500x.
Full dataset: 5.3GB -> ~15MB.

Usage:
    # Compress from raw masks directory
    python -m Implement_OVSCAN.scripts.compress_masks \
        --src_root GEN_MASK_NUSCENCES_SAM \
        --dst_root Implement_OVSCAN/data/sam3_masks

    # Decompress back to .npy
    python -m Implement_OVSCAN.scripts.compress_masks \
        --decompress \
        --src_root Implement_OVSCAN/data/sam3_masks \
        --dst_root GEN_MASK_NUSCENCES_SAM_restored
"""

import argparse
import json
import pickle
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compress_split(src_root: Path, dst_root: Path, split: str) -> dict:
    """Compress all masks in a split from .npy to .npz."""
    src_masks = src_root / split / 'masks'
    src_metadata = src_root / split / 'metadata'
    dst_masks = dst_root / split / 'masks'
    dst_metadata = dst_root / split / 'metadata'

    if not src_masks.exists():
        print(f"  Skip {split}: {src_masks} not found")
        return {'samples': 0, 'raw_bytes': 0, 'compressed_bytes': 0}

    tokens = sorted([d.name for d in src_masks.iterdir() if d.is_dir()])
    print(f"  {split}: {len(tokens)} samples")

    total_raw = 0
    total_compressed = 0

    for token in tqdm(tokens, desc=f"  Compressing {split}"):
        # Compress masks
        src_token_dir = src_masks / token
        dst_token_dir = dst_masks / token
        dst_token_dir.mkdir(parents=True, exist_ok=True)

        for npy_file in sorted(src_token_dir.glob('*.npy')):
            mask = np.load(npy_file)
            total_raw += mask.nbytes

            npz_file = dst_token_dir / (npy_file.stem + '.npz')
            np.savez_compressed(npz_file, mask=mask)
            total_compressed += npz_file.stat().st_size

        # Copy metadata JSON files as-is
        src_meta_dir = src_metadata / token
        if src_meta_dir.exists():
            dst_meta_dir = dst_metadata / token
            dst_meta_dir.mkdir(parents=True, exist_ok=True)
            for json_file in src_meta_dir.glob('*.json'):
                shutil.copy2(json_file, dst_meta_dir / json_file.name)

    # Copy index.pkl
    src_index = src_root / split / 'index.pkl'
    if src_index.exists():
        shutil.copy2(src_index, dst_root / split / 'index.pkl')

    ratio = total_raw / max(total_compressed, 1)
    print(f"  {split}: {total_raw / 1e6:.1f}MB -> {total_compressed / 1e6:.1f}MB ({ratio:.0f}x compression)")

    return {
        'samples': len(tokens),
        'raw_bytes': total_raw,
        'compressed_bytes': total_compressed,
    }


def decompress_split(src_root: Path, dst_root: Path, split: str):
    """Decompress all masks in a split from .npz to .npy."""
    src_masks = src_root / split / 'masks'
    src_metadata = src_root / split / 'metadata'
    dst_masks = dst_root / split / 'masks'
    dst_metadata = dst_root / split / 'metadata'

    if not src_masks.exists():
        print(f"  Skip {split}: {src_masks} not found")
        return

    tokens = sorted([d.name for d in src_masks.iterdir() if d.is_dir()])
    print(f"  {split}: {len(tokens)} samples")

    for token in tqdm(tokens, desc=f"  Decompressing {split}"):
        src_token_dir = src_masks / token
        dst_token_dir = dst_masks / token
        dst_token_dir.mkdir(parents=True, exist_ok=True)

        for npz_file in sorted(src_token_dir.glob('*.npz')):
            data = np.load(npz_file)
            mask = data['mask']
            npy_file = dst_token_dir / (npz_file.stem + '.npy')
            np.save(npy_file, mask)

        # Copy metadata
        src_meta_dir = src_metadata / token
        if src_meta_dir.exists():
            dst_meta_dir = dst_metadata / token
            dst_meta_dir.mkdir(parents=True, exist_ok=True)
            for json_file in src_meta_dir.glob('*.json'):
                shutil.copy2(json_file, dst_meta_dir / json_file.name)

    # Copy index.pkl
    src_index = src_root / split / 'index.pkl'
    if src_index.exists():
        shutil.copy2(src_index, dst_root / split / 'index.pkl')

    print(f"  {split}: done")


def main():
    parser = argparse.ArgumentParser(description='Compress/decompress SAM3 masks')
    parser.add_argument('--src_root', type=str, required=True,
                        help='Source mask directory')
    parser.add_argument('--dst_root', type=str, required=True,
                        help='Destination directory')
    parser.add_argument('--decompress', action='store_true',
                        help='Decompress .npz -> .npy instead of compressing')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to process')
    args = parser.parse_args()

    src = Path(args.src_root)
    dst = Path(args.dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    if args.decompress:
        print(f"Decompressing: {src} -> {dst}")
        for split in args.splits:
            decompress_split(src, dst, split)
    else:
        print(f"Compressing: {src} -> {dst}")
        stats = {}
        for split in args.splits:
            stats[split] = compress_split(src, dst, split)

        total_raw = sum(s['raw_bytes'] for s in stats.values())
        total_comp = sum(s['compressed_bytes'] for s in stats.values())
        total_samples = sum(s['samples'] for s in stats.values())
        print(f"\nTotal: {total_samples} samples, "
              f"{total_raw / 1e6:.1f}MB -> {total_comp / 1e6:.1f}MB "
              f"({total_raw / max(total_comp, 1):.0f}x)")

    # Copy class_mapping.json if it exists
    class_mapping = src / 'class_mapping.json'
    if class_mapping.exists():
        shutil.copy2(class_mapping, dst / 'class_mapping.json')


if __name__ == '__main__':
    main()
