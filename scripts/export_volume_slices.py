import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np


def _load_volume(path):
    vol = np.load(path)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {path}: {vol.shape}")
    return np.asarray(vol, dtype=np.float32)


def _slice_indices(depth, count):
    count = min(int(count), int(depth))
    return np.linspace(0, depth - 1, count, dtype=int)


def _robust_limits(vol):
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def save_grid(vol, indices, out_path, title, vmin=None, vmax=None, cmap="gray"):
    cols = min(5, len(indices))
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), dpi=160)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indices):
        ax.imshow(vol[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"z={idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration-dir", required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    iteration_dir = osp.abspath(args.iteration_dir)
    out_dir = args.out_dir or osp.join(iteration_dir, "slice_vis")
    os.makedirs(out_dir, exist_ok=True)

    pred = _load_volume(osp.join(iteration_dir, "vol_pred.npy"))
    gt_path = osp.join(iteration_dir, "vol_gt.npy")
    gt = _load_volume(gt_path) if osp.exists(gt_path) else None
    indices = _slice_indices(pred.shape[2], args.count)

    if gt is not None:
        vmin, vmax = _robust_limits(gt)
        save_grid(gt, indices, osp.join(out_dir, "vol_gt_slices.png"), "GT / FBP", vmin, vmax)
        save_grid(pred, indices, osp.join(out_dir, "vol_pred_slices.png"), "Prediction", vmin, vmax)
        diff = np.abs(pred - gt)
        _, diff_vmax = _robust_limits(diff)
        save_grid(diff, indices, osp.join(out_dir, "vol_absdiff_slices.png"), "Abs Diff", 0.0, diff_vmax, cmap="magma")
    else:
        vmin, vmax = _robust_limits(pred)
        save_grid(pred, indices, osp.join(out_dir, "vol_pred_slices.png"), "Prediction", vmin, vmax)

    print(f"Saved slice visualizations to {out_dir}")


if __name__ == "__main__":
    main()
