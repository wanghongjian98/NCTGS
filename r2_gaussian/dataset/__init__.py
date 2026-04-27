#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.utils.camera_utils import cameraList_from_camInfos
from r2_gaussian.utils.general_utils import t2a


class Scene:
    gaussians: object

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path
        self.save_volume_npy = bool(getattr(args, "save_volume_npy", True))
        self.save_volume_png = bool(getattr(args, "save_volume_png", False))
        self.save_volume_png_count = int(getattr(args, "save_volume_png_count", 15))

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        elif args.source_path.split(".")[-1] in ["yaml", "yml"]:
            # RENAF raw h5 YAML format
            scene_info = sceneLoadTypeCallbacks["RENAF_H5"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.event_data = scene_info.event_data
        self.has_gt = bool(scene_info.has_gt)
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            if self.save_volume_png:
                self._save_volume_slice_pngs(
                    point_cloud_path,
                    t2a(vol_pred),
                    t2a(vol_gt) if vol_gt is not None else None,
                    self.save_volume_png_count,
                )
            if self.save_volume_npy and vol_gt is not None:
                np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            if self.save_volume_npy:
                np.save(
                    osp.join(point_cloud_path, "vol_pred.npy"),
                    t2a(vol_pred),
                )

    def _save_volume_slice_pngs(self, point_cloud_path, vol_pred, vol_gt, count):
        import matplotlib.pyplot as plt

        def normalize_volume(vol):
            vol = np.asarray(vol, dtype=np.float32)
            if vol.ndim == 4 and vol.shape[-1] == 1:
                vol = vol[..., 0]
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got {vol.shape}.")
            return vol

        def robust_limits(vol):
            finite = vol[np.isfinite(vol)]
            if finite.size == 0:
                return 0.0, 1.0
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0
            return float(lo), float(hi)

        def save_grid(vol, indices, path, title, vmin, vmax, cmap="gray"):
            cols = min(5, len(indices))
            rows = int(np.ceil(len(indices) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), dpi=160)
            axes = np.atleast_1d(axes).reshape(rows, cols)
            for ax in axes.ravel():
                ax.axis("off")
            for ax, idx in zip(axes.ravel(), indices):
                img = ax.imshow(vol[:, :, idx], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"z={idx}", fontsize=9)
                ax.axis("off")
                fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)

        vol_pred = normalize_volume(vol_pred)
        vol_gt = normalize_volume(vol_gt) if vol_gt is not None else None
        out_dir = osp.join(point_cloud_path, "slice_vis")
        os.makedirs(out_dir, exist_ok=True)
        indices = np.linspace(0, vol_pred.shape[2] - 1, min(int(count), vol_pred.shape[2]), dtype=int)

        if vol_gt is not None:
            vmin, vmax = robust_limits(vol_gt)
            save_grid(vol_gt, indices, osp.join(out_dir, "vol_gt_slices.png"), "GT / FBP", vmin, vmax)
            save_grid(vol_pred, indices, osp.join(out_dir, "vol_pred_slices.png"), "Prediction", vmin, vmax)
            diff = np.abs(vol_pred - vol_gt)
            _, diff_vmax = robust_limits(diff)
            save_grid(diff, indices, osp.join(out_dir, "vol_absdiff_slices.png"), "Abs Diff", 0.0, diff_vmax, cmap="magma")
        else:
            vmin, vmax = robust_limits(vol_pred)
            save_grid(vol_pred, indices, osp.join(out_dir, "vol_pred_slices.png"), "Prediction", vmin, vmax)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
