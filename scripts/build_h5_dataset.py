#!/usr/bin/env python3
"""
normalize_to_h5.py

Create an HDF5 training file from an array shaped (N, 3, 128, 256)
(or (N, 128, 256, 3) if channels-last), applying robust per-sample,
per-channel min-max normalization to [-1, 1].

Usage examples:
  python build_h5_dataset.py --in data.npy --out train.h5
  python build_h5_dataset.py --in data.npz --key data --out train.h5
  python build_h5_dataset.py --in data.npy --channels-last --out train.h5
"""

import argparse
import os
import sys
import numpy as np
import h5py



def safe_log(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Elementwise natural log with positivity clamp."""
    arr = np.asarray(arr, dtype=np.float32)
    return np.log(np.clip(arr, eps, None)).astype(np.float32)

def map_per_sample_to_m11(v: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float, float]:
    """
    Robust per-sample min/max -> [-1, 1].
    Ignores NaNs/Infs; handles near-constant arrays by returning zeros.

    Returns:
      x_norm: normalized array float32
      vmin: float
      vmax: float
    """
    v = np.asarray(v, dtype=np.float32)
    m = np.isfinite(v)
    if not m.any():
        return np.zeros_like(v, dtype=np.float32), 0.0, 0.0

    vmin = float(v[m].min())
    vmax = float(v[m].max())

    if (vmax - vmin) < eps:
        return np.zeros_like(v, dtype=np.float32), vmin, vmax

    v_clipped = np.clip(v, vmin, vmax)
    x = 2.0 * (v_clipped - vmin) / (vmax - vmin) - 1.0
    return x.astype(np.float32), vmin, vmax


def load_array(path: str, key: str | None) -> np.ndarray:
    """
    Load array from .npy or .npz. For .npz, a key is required unless only one array exists.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.endswith(".npy"):
        return np.load(path)

    if path.endswith(".npz"):
        data = np.load(path)
        if key is None:
            keys = list(data.keys())
            if len(keys) == 1:
                return data[keys[0]]
            raise ValueError(f"--key is required for .npz with multiple arrays. Available keys: {keys}")
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(data.keys())}")
        return data[key]

    raise ValueError("Unsupported input format. Use .npy or .npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input .npy or .npz file")
    parser.add_argument("--key", default=None, help="Key for .npz (if needed)")
    parser.add_argument("--out", dest="out", default="train.h5", help="Output HDF5 file (default: train.h5)")
    parser.add_argument("--dataset", default="train", help="Dataset name in HDF5 (default: train)")
    #--------------log---------------------------------------
    parser.add_argument("--log-input", action="store_true",
                    help="Apply natural log to each channel BEFORE normalization (expects positive values).")
    parser.add_argument("--log-eps", type=float, default=1e-6,
                    help="Clamp minimum value before log (default: 1e-6).")
    #--------------------------------------------------------
    parser.add_argument("--channels-last", action="store_true",
                        help="Input is NHWC (N, H, W, C). Will convert to NCHW.")
    parser.add_argument("--no-minmax", action="store_true",
                        help="Do not store per-sample per-channel min/max arrays.")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"],
                        help="HDF5 compression (default: gzip)")
    parser.add_argument("--gzip-level", type=int, default=4, help="gzip compression level (default: 4)")
    parser.add_argument("--eps", type=float, default=1e-6, help="Near-constant threshold (default: 1e-6)")
    args = parser.parse_args()

    # -------- Load --------
    x = load_array(args.inp, args.key)
    x = np.asarray(x, dtype=np.float32)

    # -------- Ensure shape NCHW --------
    if args.channels_last:
        # NHWC -> NCHW
        if x.ndim != 4:
            raise ValueError(f"Expected 4D array for --channels-last. Got shape {x.shape}")
        x = np.transpose(x, (0, 3, 1, 2))

    if x.ndim != 4:
        raise ValueError(f"Expected 4D array. Got shape {x.shape}")

    N, C, H, W = x.shape
    if C != 3:
        raise ValueError(f"Expected C=3 channels. Got shape {x.shape}")
    if (H, W) != (128, 256):
        raise ValueError(f"Expected spatial size (128, 256). Got (H, W)=({H}, {W}).")

    # -------- Prepare output HDF5 --------
    if os.path.exists(args.out):
        raise FileExistsError(f"Output file already exists: {args.out} (delete it or choose another name)")

    compression = None if args.compression == "none" else args.compression
    compression_opts = args.gzip_level if compression == "gzip" else None

    with h5py.File(args.out, "w") as f:
        dset = f.create_dataset(
            args.dataset,
            shape=(N, C, H, W),
            dtype="float32",
            chunks=(1, C, H, W),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True if compression is not None else False,
        )

        if not args.no_minmax:
            dmin = f.create_dataset("train_min", shape=(N, C), dtype="float32")
            dmax = f.create_dataset("train_max", shape=(N, C), dtype="float32")
        else:
            dmin = dmax = None

        # -------- Normalize + write (stream sample-by-sample) --------
        for i in range(N):
            x_i = x[i]  # (3, 128, 256)
            out_i = np.empty_like(x_i, dtype=np.float32)

            for c in range(C):
                v = x_i[c]
                if args.log_input:
                    v = safe_log(v, eps=args.log_eps)
                norm_c, vmin, vmax = map_per_sample_to_m11(v, eps=args.eps)

                out_i[c] = norm_c
                if dmin is not None:
                    dmin[i, c] = vmin
                    dmax[i, c] = vmax

            dset[i] = out_i

            if (i + 1) % 50 == 0 or (i + 1) == N:
                print(f"Wrote {i+1}/{N} samples", flush=True)

    print(f"\nDone. Saved normalized dataset to: {args.out}")
    print(f"Dataset name: {args.dataset}")
    if not args.no_minmax:
        print("Also saved: train_min, train_max (shape N x C)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
