import os
import sys
from typing import NamedTuple
import numpy as np
import os.path as osp
import json
import torch
import pickle
import yaml

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import BasicPointCloud, fetchPly
from r2_gaussian.dataset.raw_h5_reader import (
    centered_crop_offset,
    load_event_csv,
    load_h5_theta_degrees,
    load_projection_subset,
    repair_leading_zero_angles_deg,
)
from r2_gaussian.utils.cfg_utils import load_config

mode_id = {
    "parallel": 0,
    "cone": 1,
}


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    angle: float
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mode: int
    scanner_cfg: dict


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor
    scanner_cfg: dict
    scene_scale: float
    event_data: object = None
    has_gt: bool = True


def readBlenderInfo(path, eval):
    """Read blender format CT data."""
    # Read meta data
    meta_data_path = osp.join(path, "meta_data.json")
    with open(meta_data_path, "r") as handle:
        meta_data = json.load(handle)
    meta_data["vol"] = osp.join(path, meta_data["vol"])

    if not "dVoxel" in meta_data["scanner"]:
        meta_data["scanner"]["dVoxel"] = list(
            np.array(meta_data["scanner"]["sVoxel"])
            / np.array(meta_data["scanner"]["nVoxel"])
        )
    if not "dDetector" in meta_data["scanner"]:
        meta_data["scanner"]["dDetector"] = list(
            np.array(meta_data["scanner"]["sDetector"])
            / np.array(meta_data["scanner"]["nDetector"])
        )

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        meta_data["scanner"][key_to_scale] = (
            np.array(meta_data["scanner"][key_to_scale]) * scene_scale
        ).tolist()

    cam_infos = readCTameras(meta_data, path, eval, scene_scale)
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]

    vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float().cuda()

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=meta_data["scanner"],
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


def readCTameras(meta_data, source_path, eval=False, scene_scale=1.0):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    cam_infos = {"train": [], "test": []}
    for split in splits:
        split_info = meta_data["proj_" + split]
        n_split = len(split_info)
        if split == "test":
            uid_offset = len(meta_data["proj_train"])
        else:
            uid_offset = 0
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_info = meta_data["proj_" + split][i_split]
            frame_angle = frame_info["angle"]

            # CT 'transform_matrix' is a camera-to-world transform
            c2w = angle2pose(cam_cfg["DSO"], frame_angle)  # c2w
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = osp.join(source_path, frame_info["file_path"])
            image = np.load(image_path) * scene_scale
            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
            FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2

            mode = mode_id[cam_cfg["mode"]]

            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=osp.basename(image_path).split(".")[0],
                width=cam_cfg["nDetector"][1],
                height=cam_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=cam_cfg,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")
    return cam_infos


def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform


def readNAFInfo(path, eval):
    """Read blender format CT data."""
    # Read data
    with open(path, "rb") as f:
        data = pickle.load(f)
    # ! NAF scanner are measured in mm, but projections are measured in m. Therefore we need to / 1000.
    scanner_cfg = {
        "DSD": data["DSD"] / 1000,
        "DSO": data["DSO"] / 1000,
        "nVoxel": data["nVoxel"],
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),
        "nDetector": data["nDetector"],
        "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),
        "sDetector": (
            np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000
        ).tolist(),
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),
        "totalAngle": data["totalAngle"],
        "startAngle": data["startAngle"],
        "accuracy": data["accuracy"],
        "mode": data["mode"],
        "filter": None,
    }

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        scanner_cfg[key_to_scale] = (
            np.array(scanner_cfg[key_to_scale]) * scene_scale
        ).tolist()

    # Generate camera infos
    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]
    cam_infos = {"train": [], "test": []}
    for split in splits:
        if split == "test":
            uid_offset = data["numTrain"]
            n_split = data["numVal"]
        else:
            uid_offset = 0
            n_split = data["numTrain"]
        if split == "test" and "val" in data:
            data_split = data["val"]
        else:
            data_split = data[split]
        angles = data_split["angles"]
        projs = data_split["projections"]

        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_angle = angles[i_split]
            c2w = angle2pose(scanner_cfg["DSO"], frame_angle)
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image = projs[i_split] * scene_scale

            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
            FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2

            mode = mode_id[scanner_cfg["mode"]]

            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=None,
                image_name=f"{i_split + uid_offset:04d}",
                width=scanner_cfg["nDetector"][1],
                height=scanner_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")

    # Store other data
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    vol_gt = torch.from_numpy(data["image"]).float().cuda()
    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=scanner_cfg,
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


def _renaf_scanner_cfg(geometry_cfg):
    n_voxel_full = list(geometry_cfg["nVoxel"])
    n_voxel = list(n_voxel_full)
    if geometry_cfg.get("voxel_slice_count_z") is not None:
        n_voxel[2] = int(geometry_cfg["voxel_slice_count_z"])

    d_voxel = np.array(geometry_cfg["dVoxel"], dtype=np.float64) / 1000.0
    renaf_n_detector = list(geometry_cfg["nDetector"])  # RENAF config is [W, H].
    renaf_d_detector = np.array(geometry_cfg["dDetector"], dtype=np.float64) / 1000.0
    n_detector = [int(renaf_n_detector[1]), int(renaf_n_detector[0])]  # R2 uses [H, W].
    d_detector = [float(renaf_d_detector[1]), float(renaf_d_detector[0])]
    scanner_cfg = {
        "DSD": float(geometry_cfg["DSD"]) / 1000.0,
        "DSO": float(geometry_cfg["DSO"]) / 1000.0,
        "nVoxel": n_voxel,
        "dVoxel": d_voxel.tolist(),
        "sVoxel": (np.array(n_voxel, dtype=np.float64) * d_voxel).tolist(),
        "nDetector": n_detector,
        "dDetector": d_detector,
        "sDetector": (np.array(n_detector, dtype=np.float64) * np.array(d_detector)).tolist(),
        "offOrigin": (np.array(geometry_cfg.get("offOrigin", [0.0, 0.0, 0.0])) / 1000.0).tolist(),
        "offDetector": (np.array(geometry_cfg.get("offDetector", [0.0, 0.0])) / 1000.0).tolist(),
        "totalAngle": float(geometry_cfg.get("totalAngle", np.pi)),
        "startAngle": float(geometry_cfg.get("startAngle", 0.0)),
        "accuracy": float(geometry_cfg.get("accuracy", 0.5)),
        "mode": geometry_cfg.get("mode", "parallel"),
        "filter": None,
    }
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        scanner_cfg[key_to_scale] = (np.array(scanner_cfg[key_to_scale]) * scene_scale).tolist()
    return scanner_cfg, scene_scale


def _read_renaf_angles(raw_cfg, proj_select):
    h5_path = raw_cfg["h5_path"]
    projection_start_index = int(raw_cfg.get("projection_start_index", 0))
    theta_dataset = raw_cfg.get("theta_dataset", "/exchange/theta")
    angles_full_deg = load_h5_theta_degrees(h5_path, theta_dataset)
    angles_full_deg = repair_leading_zero_angles_deg(angles_full_deg[projection_start_index:])
    angles_full_deg = angles_full_deg - angles_full_deg[0]
    index_len_180 = int(np.sum(angles_full_deg < 180.0))
    if index_len_180 <= 0:
        raise ValueError("No forward-sweep projections (< 180 deg) were found.")
    proj_select = min(int(proj_select), index_len_180)
    relative_indices = np.linspace(0, index_len_180 - 1, num=proj_select, dtype=int)
    h5_indices = relative_indices + projection_start_index
    angles = np.deg2rad(angles_full_deg[relative_indices]).astype(np.float32, copy=False)
    return angles, h5_indices, index_len_180


def _resolve_fbp_detector_hw(raw_cfg, scanner_cfg):
    detector_size = raw_cfg.get("fbp_detector_size")
    if detector_size is None:
        return int(scanner_cfg["nDetector"][0]), int(scanner_cfg["nDetector"][1])
    if len(detector_size) != 2:
        raise ValueError(
            "raw_data.fbp_detector_size must be [width, height] in RENAF convention."
        )
    return int(detector_size[1]), int(detector_size[0])


def _fbp_scanner_cfg(scanner_cfg, detector_h, detector_w):
    cfg = dict(scanner_cfg)
    cfg["nDetector"] = [int(detector_h), int(detector_w)]
    cfg["sDetector"] = (
        np.array(cfg["nDetector"], dtype=np.float64)
        * np.array(cfg["dDetector"], dtype=np.float64)
    ).tolist()
    return cfg


def _renaf_fbp_cache_path(raw_cfg, scanner_cfg, index_len_180):
    n_voxel_tag = "x".join(str(int(v)) for v in scanner_cfg["nVoxel"])
    detector_h, detector_w = _resolve_fbp_detector_hw(raw_cfg, scanner_cfg)
    n_detector_tag = f"{detector_h}x{detector_w}"
    align_mode = str(raw_cfg.get("projection_align_mode", "affine")).strip().lower()
    align_tag = f"_align{align_mode}" if align_mode != "affine" else ""
    pad_h, pad_w = _parse_fbp_pad(raw_cfg)
    pad_tag = f"_pad{pad_h}x{pad_w}" if pad_h > 0 or pad_w > 0 else ""
    return (
        f"{raw_cfg['h5_path']}.r2_fbp_"
        f"n{int(index_len_180)}_v{n_voxel_tag}_d{n_detector_tag}{align_tag}{pad_tag}.npy"
    )


def _parse_fbp_pad(raw_cfg):
    pad = raw_cfg.get("fbp_pad_detector", [0, 0])
    if isinstance(pad, (int, float)):
        pad_h = pad_w = int(pad)
    elif isinstance(pad, dict):
        pad_h = int(pad.get("h", pad.get("height", 0)))
        pad_w = int(pad.get("w", pad.get("width", 0)))
    else:
        if len(pad) != 2:
            raise ValueError(
                "raw_data.fbp_pad_detector must be a scalar or [pad_h, pad_w]."
            )
        pad_h, pad_w = int(pad[0]), int(pad[1])
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"fbp_pad_detector must be non-negative, got {[pad_h, pad_w]}.")
    return pad_h, pad_w


def _pad_fbp_projections(projs, scanner_cfg, raw_cfg):
    pad_h, pad_w = _parse_fbp_pad(raw_cfg)
    if pad_h == 0 and pad_w == 0:
        return projs, scanner_cfg

    pad_mode = str(raw_cfg.get("fbp_pad_mode", "edge")).strip().lower()
    np_pad_mode = pad_mode
    pad_kwargs = {}
    if pad_mode == "constant":
        pad_kwargs["constant_values"] = float(raw_cfg.get("fbp_pad_constant", 0.0))
    elif pad_mode not in ["edge", "reflect", "symmetric"]:
        raise ValueError(
            "raw_data.fbp_pad_mode must be one of edge, reflect, symmetric, constant; "
            f"got {pad_mode!r}."
        )

    padded = np.pad(
        projs,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode=np_pad_mode,
        **pad_kwargs,
    )
    padded_cfg = dict(scanner_cfg)
    n_detector = np.array(scanner_cfg["nDetector"], dtype=np.int64).copy()
    n_detector += np.array([2 * pad_h, 2 * pad_w], dtype=np.int64)
    padded_cfg["nDetector"] = n_detector.tolist()
    padded_cfg["sDetector"] = (
        n_detector.astype(np.float64) * np.array(scanner_cfg["dDetector"], dtype=np.float64)
    ).tolist()
    print(
        "[RENAF h5] padded FBP projections: "
        f"pad_h={pad_h}, pad_w={pad_w}, mode={pad_mode}, "
        f"shape={projs.shape}->{padded.shape}"
    )
    return np.ascontiguousarray(padded), padded_cfg


def _load_or_build_renaf_fbp_gt(raw_cfg, scanner_cfg, scene_scale, index_len_180):
    import copy

    import torch

    from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume

    cache_path = _renaf_fbp_cache_path(raw_cfg, scanner_cfg, index_len_180)
    if os.path.exists(cache_path):
        vol = np.load(cache_path, mmap_mode=None).astype(np.float32, copy=False)
        print(f"[RENAF h5] loaded cached FBP GT: {cache_path}, shape={vol.shape}")
        return torch.from_numpy(vol).float().cuda()

    all_indices = np.arange(index_len_180, dtype=np.int64) + int(
        raw_cfg.get("projection_start_index", 0)
    )
    angles_full_deg = load_h5_theta_degrees(
        raw_cfg["h5_path"], raw_cfg.get("theta_dataset", "/exchange/theta")
    )
    projection_start_index = int(raw_cfg.get("projection_start_index", 0))
    angles_full_deg = repair_leading_zero_angles_deg(
        angles_full_deg[projection_start_index:]
    )
    angles_full_deg = angles_full_deg - angles_full_deg[0]
    angles_all = np.deg2rad(angles_full_deg[:index_len_180]).astype(np.float32)
    detector_h, detector_w = _resolve_fbp_detector_hw(raw_cfg, scanner_cfg)
    fbp_base_scanner_cfg = _fbp_scanner_cfg(scanner_cfg, detector_h, detector_w)
    print(
        "[RENAF h5] building FBP GT from all projections: "
        f"n={index_len_180}, detector=[{detector_w}, {detector_h}]"
    )
    projs_all = load_projection_subset(
        raw_cfg["h5_path"],
        raw_cfg,
        all_indices,
        detector_h=int(detector_h),
        detector_w=int(detector_w),
    )
    projs_all = projs_all * float(scene_scale)
    projs_all, fbp_scanner_cfg = _pad_fbp_projections(
        projs_all, fbp_base_scanner_cfg, raw_cfg
    )
    geo = get_geometry_tigre(fbp_scanner_cfg)
    vol = recon_volume(projs_all, angles_all, copy.deepcopy(geo), "fbp").astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, vol)
    print(f"[RENAF h5] saved FBP GT: {cache_path}, shape={vol.shape}")
    return torch.from_numpy(vol).float().cuda()


def _build_renaf_events(raw_cfg, scanner_cfg, scene_scale, index_len_180):
    events_path = (
        raw_cfg.get("events_csv_path")
        or raw_cfg.get("csv_path")
        or raw_cfg.get("events_path")
    )
    if not events_path:
        return None

    events = load_event_csv(
        events_path,
        column_order=raw_cfg.get("events_csv_columns", ("x", "y", "p", "t")),
        use_cache=bool(raw_cfg.get("cache_events", True)),
    )
    event_sensor_size = raw_cfg.get("event_sensor_size", [1280, 720])
    event_raw_w, event_raw_h = int(event_sensor_size[0]), int(event_sensor_size[1])
    detector_h, detector_w = int(scanner_cfg["nDetector"][0]), int(scanner_cfg["nDetector"][1])
    event_to_frame_affine = np.array(
        raw_cfg.get(
            "event_to_frame_affine",
            [[1.0927234, -0.00763725, -63.842716], [0.00763725, 1.0920073, -3.5674274]],
        ),
        dtype=np.float32,
    )
    crop_x = centered_crop_offset(event_raw_w, detector_w)
    crop_y = centered_crop_offset(event_raw_h, detector_h)
    center_raw = np.array([crop_x + 0.5 * (detector_w - 1), crop_y + 0.5 * (detector_h - 1)])
    center_without_scale = np.array([0.5 * (detector_w - 1), 0.5 * (detector_h - 1)])
    translation = event_to_frame_affine[:, :2].astype(np.float64) @ center_raw + event_to_frame_affine[:, 2] - center_without_scale

    exposure_time_us = int(raw_cfg.get("exposure_time_us", 10000))
    event_start_frame_index = int(raw_cfg.get("event_start_frame_index", 3))
    total_len = int(index_len_180 * exposure_time_us)
    t_start = int(events["t"][0] + event_start_frame_index * exposure_time_us)
    t_end = int(t_start + total_len)
    valid_time = (events["t"] >= t_start) & (events["t"] <= t_end)
    events = events[valid_time]

    x_raw = (event_raw_w - 1) - events["x"].astype(np.int32, copy=False)
    y_raw = events["y"].astype(np.int32, copy=False)
    x_aligned = np.clip(x_raw.astype(np.float32) - crop_x + translation[0], 0.0, detector_w - 1.0)
    y_aligned = np.clip(y_raw.astype(np.float32) - crop_y + translation[1], 0.0, detector_h - 1.0)
    x = np.rint(x_aligned).astype(np.int64)
    y = np.rint(y_aligned).astype(np.int64)

    center_y_half_width = raw_cfg.get("center_y_half_width")
    if center_y_half_width is not None:
        half = int(center_y_half_width)
        y_min = max(0, detector_h // 2 - half)
        y_max = min(detector_h - 1, detector_h // 2 + half)
    else:
        y_min, y_max = 0, detector_h - 1
    valid_xy = (x >= 0) & (x < detector_w) & (y >= y_min) & (y <= y_max)

    prev_ts = {}
    sample_x, sample_y, sample_t0, sample_t1, sample_p = [], [], [], [], []
    pixel_id = y * detector_w + x
    for idx in range(events.shape[0]):
        if not bool(valid_xy[idx]):
            continue
        pid = int(pixel_id[idx])
        t_curr = int(events["t"][idx])
        t_prev = prev_ts.get(pid)
        if t_prev is not None and t_curr > t_prev:
            sample_x.append(int(x[idx]))
            sample_y.append(int(y[idx]))
            sample_t0.append(int(t_prev))
            sample_t1.append(t_curr)
            sample_p.append(1 if int(events["p"][idx]) > 0 else -1)
        prev_ts[pid] = t_curr

    print(
        "[RENAF events] "
        f"window_us=[{t_start}, {t_end}], raw_kept={int(np.count_nonzero(valid_time))}, "
        f"pairs={len(sample_x)}"
    )
    angle_range_deg = raw_cfg.get("event_angle_range_deg", [0.0, 180.0])
    return {
        "x": np.asarray(sample_x, dtype=np.int16),
        "y": np.asarray(sample_y, dtype=np.int16),
        "start_ts": np.asarray(sample_t0, dtype=np.int64),
        "end_ts": np.asarray(sample_t1, dtype=np.int64),
        "polarity": np.asarray(sample_p, dtype=np.int8),
        "t_start": t_start,
        "total_len": total_len,
        "angle_min": float(np.deg2rad(angle_range_deg[0])),
        "angle_max": float(np.deg2rad(angle_range_deg[1])),
        "scene_scale": float(scene_scale),
        "detector_width": detector_w,
        "detector_height": detector_h,
    }


def readRENAFH5Info(path, eval):
    cfg = load_config(path)
    raw_cfg = dict(cfg.get("exp", {}).get("raw_data", {}))
    geometry_cfg = dict(cfg.get("geometry", {}))
    train_cfg = dict(cfg.get("train", {}))
    if not raw_cfg.get("h5_path"):
        raise ValueError("RENAF YAML source requires exp.raw_data.h5_path.")

    scanner_cfg, scene_scale = _renaf_scanner_cfg(geometry_cfg)
    detector_h, detector_w = scanner_cfg["nDetector"]
    proj_select = int(train_cfg.get("proj_num", raw_cfg.get("proj_select", 450)))
    angles, h5_indices, index_len_180 = _read_renaf_angles(raw_cfg, proj_select)
    projs = load_projection_subset(
        raw_cfg["h5_path"],
        raw_cfg,
        h5_indices,
        detector_h=int(detector_h),
        detector_w=int(detector_w),
    )
    projs = projs * float(scene_scale)
    print(
        "[RENAF h5] loaded projections: "
        f"shape={projs.shape}, angle_count_180={index_len_180}, scene_scale={scene_scale:.6g}"
    )

    n_train = projs.shape[0]
    n_test = 0
    if eval and n_train > 8:
        n_test = max(1, n_train // 10)
    train_indices = np.arange(0, n_train - n_test, dtype=np.int64)
    test_indices = np.arange(n_train - n_test, n_train, dtype=np.int64) if n_test else []

    def make_cam_info(i_split, uid):
        frame_angle = float(angles[i_split])
        c2w = angle2pose(scanner_cfg["DSO"], frame_angle)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        fov_x = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
        fov_y = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2
        return CameraInfo(
            uid=uid,
            R=R,
            T=T,
            angle=frame_angle,
            FovY=fov_y,
            FovX=fov_x,
            image=projs[i_split],
            image_path=None,
            image_name=f"{uid:04d}",
            width=scanner_cfg["nDetector"][1],
            height=scanner_cfg["nDetector"][0],
            mode=mode_id[scanner_cfg["mode"]],
            scanner_cfg=scanner_cfg,
        )

    train_cam_infos = [make_cam_info(int(i), uid) for uid, i in enumerate(train_indices)]
    test_cam_infos = [
        make_cam_info(int(i), len(train_cam_infos) + uid) for uid, i in enumerate(test_indices)
    ]
    source_train_mode = str(train_cfg.get("mode", "proj")).strip()
    event_data = None
    if source_train_mode in ["event", "proj_event"]:
        event_data = _build_renaf_events(raw_cfg, scanner_cfg, scene_scale, index_len_180)
    vol_gt = None
    if bool(raw_cfg.get("build_fbp_gt", True)):
        vol_gt = _load_or_build_renaf_fbp_gt(
            raw_cfg, scanner_cfg, scene_scale, index_len_180
        )

    return SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=scanner_cfg,
        vol=vol_gt,
        scene_scale=scene_scale,
        event_data=event_data,
        has_gt=vol_gt is not None,
    )


sceneLoadTypeCallbacks = {
    "Blender": readBlenderInfo,
    "NAF": readNAFInfo,
    "RENAF_H5": readRENAFH5Info,
}
