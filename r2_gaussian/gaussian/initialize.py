import os
import sys
import os.path as osp
from glob import glob
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.graphics_utils import fetchPly
from r2_gaussian.utils.system_utils import searchForMaxIteration
from r2_gaussian.utils.cfg_utils import load_config


def _create_random_yaml_init(args, ply_path, n_points=50000):
    cfg = load_config(args.source_path)
    geometry_cfg = cfg["geometry"]
    n_voxel = list(geometry_cfg["nVoxel"])
    if geometry_cfg.get("voxel_slice_count_z") is not None:
        n_voxel[2] = int(geometry_cfg["voxel_slice_count_z"])
    d_voxel = np.array(geometry_cfg["dVoxel"], dtype=np.float64) / 1000.0
    s_voxel = (np.array(n_voxel, dtype=np.float64) * d_voxel).tolist()
    scene_scale = 2 / max(s_voxel)
    scanner_cfg = {
        "offOrigin": (
            np.array(geometry_cfg.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float64)
            / 1000.0
            * scene_scale
        ).tolist(),
        "sVoxel": (np.array(s_voxel, dtype=np.float64) * scene_scale).tolist(),
    }
    rng = np.random.default_rng(0)
    positions = np.array(scanner_cfg["offOrigin"])[None, :] + np.array(
        scanner_cfg["sVoxel"]
    )[None, :] * (rng.random((int(n_points), 3)) - 0.5)
    densities = rng.random((int(n_points), 1)) * 0.05
    os.makedirs(osp.dirname(ply_path), exist_ok=True)
    np.save(ply_path, np.concatenate([positions, densities], axis=-1).astype(np.float32))
    print(f"Created random RENAF initialization point cloud: {ply_path}")


def _create_fbp_yaml_init(args, ply_path, n_points=50000):
    cfg = load_config(args.source_path)
    raw_cfg = cfg.get("exp", {}).get("raw_data", {})
    geometry_cfg = cfg["geometry"]
    h5_path = raw_cfg.get("h5_path")
    if not h5_path:
        return False

    candidates = sorted(glob(f"{h5_path}.r2_fbp_*.npy"))
    align_mode = str(raw_cfg.get("projection_align_mode", "affine")).strip().lower()
    if align_mode != "affine":
        align_tag = f"_align{align_mode}"
        candidates = [path for path in candidates if align_tag in osp.basename(path)]
    if not candidates:
        return False

    vol_path = candidates[-1]
    vol = np.load(vol_path, mmap_mode=None).astype(np.float32, copy=False)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.clip(vol, 0.0, None)
    scale_ref = float(np.percentile(vol[vol > 0], 99.0)) if np.any(vol > 0) else 1.0
    if not np.isfinite(scale_ref) or scale_ref <= 0:
        scale_ref = 1.0
    vol = np.clip(vol / scale_ref, 0.0, 1.0)
    n_voxel = list(geometry_cfg["nVoxel"])
    if geometry_cfg.get("voxel_slice_count_z") is not None:
        n_voxel[2] = int(geometry_cfg["voxel_slice_count_z"])
    d_voxel = np.array(geometry_cfg["dVoxel"], dtype=np.float64) / 1000.0
    s_voxel = np.array(n_voxel, dtype=np.float64) * d_voxel
    scene_scale = 2 / max(s_voxel)
    d_voxel = d_voxel * scene_scale
    s_voxel = s_voxel * scene_scale
    off_origin = (
        np.array(geometry_cfg.get("offOrigin", [0.0, 0.0, 0.0]), dtype=np.float64)
        / 1000.0
        * scene_scale
    )

    thresh = float(raw_cfg.get("fbp_init_density_thresh", 0.02))
    valid_indices = np.argwhere(np.isfinite(vol) & (vol > thresh))
    if valid_indices.shape[0] < int(n_points):
        flat = vol.reshape(-1)
        finite = np.isfinite(flat)
        if np.count_nonzero(finite) == 0:
            return False
        take = min(int(n_points), int(np.count_nonzero(finite)))
        top_flat = np.argpartition(flat[finite], -take)[-take:]
        finite_indices = np.flatnonzero(finite)
        valid_indices = np.column_stack(np.unravel_index(finite_indices[top_flat], vol.shape))

    rng = np.random.default_rng(0)
    replace = valid_indices.shape[0] < int(n_points)
    sampled_indices = valid_indices[
        rng.choice(valid_indices.shape[0], int(n_points), replace=replace)
    ]
    sampled_positions = sampled_indices * d_voxel - s_voxel / 2 + off_origin
    sampled_densities = vol[
        sampled_indices[:, 0],
        sampled_indices[:, 1],
        sampled_indices[:, 2],
    ]
    sampled_densities = np.clip(sampled_densities, 1e-4, 1.0)
    sampled_densities = sampled_densities * float(raw_cfg.get("fbp_init_density_rescale", 0.2))
    sampled_densities = np.clip(sampled_densities, 1e-4, None)
    os.makedirs(osp.dirname(ply_path), exist_ok=True)
    np.save(
        ply_path,
        np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1).astype(np.float32),
    )
    print(f"Created FBP RENAF initialization point cloud from {vol_path}: {ply_path}")
    return True


def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            elif args.source_path.split(".")[-1] in ["yaml", "yml"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).rsplit(".", 1)[0] + ".npy",
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path

        if (
            not osp.exists(ply_path)
            and args.source_path.split(".")[-1] in ["yaml", "yml"]
        ):
            init_num_points = int(getattr(args, "init_num_points", 50000))
            if not _create_fbp_yaml_init(args, ply_path, n_points=init_num_points):
                _create_random_yaml_init(args, ply_path, n_points=init_num_points)

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        gaussians.create_from_pcd(xyz, density, 1.0)

    return loaded_iter
