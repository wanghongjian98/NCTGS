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
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.dataset.dataset_readers import angle2pose, mode_id
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice


class SimpleEventParams(nn.Module):
    def __init__(self, tau_us=300.0, tau_min_us=1.0, tau_max_us=1000.0):
        super().__init__()
        self.register_buffer("tau_min_us", torch.tensor(float(tau_min_us), dtype=torch.float32))
        self.register_buffer("tau_max_us", torch.tensor(float(tau_max_us), dtype=torch.float32))
        self.register_buffer("tau_us", torch.tensor(float(tau_us), dtype=torch.float32))

    def get_tau_us(self):
        return self.tau_us.clamp(float(self.tau_min_us), float(self.tau_max_us))


def make_event_camera(scanner_cfg, angle, uid, data_device):
    c2w = angle2pose(scanner_cfg["DSO"], float(angle))
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]
    fov_x = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
    fov_y = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2
    height = int(scanner_cfg["nDetector"][1])
    width = int(scanner_cfg["nDetector"][0])
    image = torch.zeros((1, height, width), dtype=torch.float32)
    return Camera(
        colmap_id=uid,
        scanner_cfg=scanner_cfg,
        R=R,
        T=T,
        angle=float(angle),
        mode=mode_id[scanner_cfg["mode"]],
        FoVx=fov_x,
        FoVy=fov_y,
        image=image,
        image_name=f"event_{uid:04d}",
        uid=uid,
        data_device=data_device,
    )


def event_timestamps_to_angles(ts_us, event_data, angle_bins):
    alpha = (ts_us.astype(np.float64) - float(event_data["t_start"])) / float(event_data["total_len"])
    alpha = np.clip(alpha, 0.0, 1.0)
    if int(angle_bins) > 1:
        alpha = np.round(alpha * (int(angle_bins) - 1)) / float(int(angle_bins) - 1)
    return (
        float(event_data["angle_min"])
        + alpha * (float(event_data["angle_max"]) - float(event_data["angle_min"]))
    ).astype(np.float32)


def render_event_values(angles, xs, ys, scene, gaussians, pipe, data_device):
    values = torch.empty((angles.shape[0],), dtype=torch.float32, device="cuda")
    unique_angles, inverse = np.unique(angles, return_inverse=True)
    for i_angle, angle in enumerate(unique_angles):
        mask = inverse == i_angle
        camera = make_event_camera(scene.scanner_cfg, float(angle), int(i_angle), data_device)
        image = render(camera, gaussians, pipe)["render"][0]
        x_t = torch.from_numpy(xs[mask].astype(np.int64, copy=False)).long().cuda()
        y_t = torch.from_numpy(ys[mask].astype(np.int64, copy=False)).long().cuda()
        idx_t = torch.from_numpy(np.flatnonzero(mask).astype(np.int64)).long().cuda()
        values[idx_t] = image[y_t, x_t]
    return values


def compute_event_loss(scene, gaussians, pipe, opt, event_params, data_device):
    event_data = scene.event_data
    total_pairs = int(event_data["x"].shape[0])
    if total_pairs == 0:
        return None

    sample_n = min(int(opt.event_pairs_per_iter), total_pairs)
    indices = np.random.choice(total_pairs, size=sample_n, replace=total_pairs < sample_n)
    xs = event_data["x"][indices].astype(np.int64, copy=False)
    ys = event_data["y"][indices].astype(np.int64, copy=False)
    t0 = event_data["start_ts"][indices].astype(np.float64, copy=False)
    t1 = event_data["end_ts"][indices].astype(np.float64, copy=False)
    polarity_np = event_data["polarity"][indices].astype(np.float32, copy=False)
    tau_us = float(event_params.get_tau_us().detach().cpu().item())
    t_ref = t0 + tau_us
    valid = t_ref < t1
    if not np.any(valid):
        return torch.tensor(0.0, device="cuda")

    xs = xs[valid]
    ys = ys[valid]
    t_ref = t_ref[valid]
    t1 = t1[valid]
    polarity = torch.from_numpy(polarity_np[valid]).float().cuda()
    ref_angles = event_timestamps_to_angles(t_ref, event_data, opt.event_angle_bins)
    curr_angles = event_timestamps_to_angles(t1, event_data, opt.event_angle_bins)
    log_l_ref = -render_event_values(ref_angles, xs, ys, scene, gaussians, pipe, data_device)
    log_l_curr = -render_event_values(curr_angles, xs, ys, scene, gaussians, pipe, data_device)
    signed_diff = polarity * (log_l_curr - log_l_ref)
    beta = float(opt.event_softplus_beta)
    return (F.softplus(-beta * signed_diff) / beta).mean()


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    event_params = None
    if scene.event_data is not None and opt.event_lambda > 0:
        event_params = SimpleEventParams(
            tau_us=opt.event_tau_us,
            tau_min_us=opt.event_tau_min_us,
            tau_max_us=opt.event_tau_max_us,
        ).cuda()
        print(
            "[Events] enabled: "
            f"pairs={scene.event_data['x'].shape[0]}, "
            f"pairs_per_iter={opt.event_pairs_per_iter}, "
            f"angle_bins={opt.event_angle_bins}, "
            f"lambda={opt.event_lambda}, tau_us={float(event_params.get_tau_us().item()):.3f}"
        )
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
        if event_params is not None and iteration >= opt.event_start_iter:
            loss_event = compute_event_loss(
                scene, gaussians, pipe, opt, event_params, dataset.data_device
            )
            if loss_event is not None:
                loss["event"] = loss_event
                loss["total"] = loss["total"] + opt.event_lambda * loss_event

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        if scene.vol_gt is not None:
            vol_pred = queryFunc(scene.gaussians)["vol"]
            vol_gt = scene.vol_gt
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
            eval_dict = {
                "psnr_3d": psnr_3d,
                "ssim_3d": ssim_3d,
                "ssim_3d_x": ssim_3d_axis[0],
                "ssim_3d_y": ssim_3d_axis[1],
                "ssim_3d_z": ssim_3d_axis[2],
            }
            with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
            if tb_writer:
                image_show_3d = np.concatenate(
                    [
                        show_two_slice(
                            vol_gt[..., i],
                            vol_pred[..., i],
                            f"slice {i} gt",
                            f"slice {i} pred",
                            vmin=vol_gt[..., i].min(),
                            vmax=vol_gt[..., i].max(),
                            save=True,
                        )
                        for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                    ],
                    axis=0,
                )
                image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                tb_writer.add_images(
                    "reconstruction/slice-gt_pred_diff",
                    image_show_3d,
                    global_step=iteration,
                )
                tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
                tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
            tqdm.write(
                f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
            )
        else:
            with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                yaml.dump({"skipped": "no ground-truth volume"}, f, default_flow_style=False)
            tqdm.write(
                f"[ITER {iteration}] Evaluating: no 3D GT, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
            )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
