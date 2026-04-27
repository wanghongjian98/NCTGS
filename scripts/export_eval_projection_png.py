import os
import os.path as osp
import sys
from argparse import ArgumentParser

import matplotlib
import numpy as np
import torch

sys.path.append("./")

from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, render
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.plot_utils import show_two_slice


def limit_cameras(cameras, max_cameras):
    max_cameras = int(max_cameras)
    if max_cameras <= 0 or len(cameras) <= max_cameras:
        return cameras
    indices = np.linspace(0, len(cameras) - 1, max_cameras, dtype=int)
    return [cameras[int(i)] for i in indices]


def save_grid(rows, save_path):
    if not rows:
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, np.concatenate(rows, axis=0))


@torch.no_grad()
def export_split(scene, pipe, split_name, cameras, save_dir, count):
    cameras = limit_cameras(cameras, count)
    if len(cameras) == 0:
        return None

    rows = []
    for viewpoint in cameras:
        image = render(viewpoint, scene.gaussians, pipe)["render"]
        gt_image = viewpoint.original_image.to("cuda")
        rows.append(
            show_two_slice(
                gt_image[0],
                image[0],
                f"{viewpoint.image_name} gt",
                f"{viewpoint.image_name} render",
                vmin=gt_image[0].min(),
                vmax=gt_image[0].max(),
                save=True,
            )
        )

    save_path = osp.join(save_dir, f"eval2d_{split_name}_projections.png")
    save_grid(rows, save_path)
    return save_path


def main():
    parser = ArgumentParser(description="Export GT/render projection comparison PNGs.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)
    args_dict = vars(args)
    for key in list(cfg.keys()):
        args_dict[key] = cfg[key]

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    model_params, ckpt_iter = torch.load(args.checkpoint)
    gaussians.restore(model_params, opt)
    scene.gaussians = gaussians

    iteration = int(args.iteration or ckpt_iter)
    save_dir = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")

    train_path = export_split(
        scene, pipe, "render_train", scene.getTrainCameras(), save_dir, args.count
    )
    test_path = export_split(
        scene, pipe, "render_test", scene.getTestCameras(), save_dir, args.count
    )
    for path in [train_path, test_path]:
        if path:
            print(path)


if __name__ == "__main__":
    main()
