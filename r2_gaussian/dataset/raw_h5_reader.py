import os
import re
import subprocess
import tempfile
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


RAW_EVENT_DTYPE = np.dtype(
    [
        ("t", np.int64),
        ("x", np.int32),
        ("y", np.int32),
        ("p", np.int16),
    ]
)


def _run_h5dump(path: str, dataset: str, header_only: bool = False) -> str:
    cmd = ["h5dump"]
    if header_only:
        cmd.append("-H")
    cmd.extend(["-d", dataset, path])
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("h5dump is required to read the raw h5 file.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"h5dump failed for dataset {dataset!r} in file {path!r}: {exc.stderr}"
        ) from exc
    return proc.stdout


def get_h5_dataset_shape(path: str, dataset: str) -> Sequence[int]:
    dump = _run_h5dump(path, dataset, header_only=True)
    match = re.search(r"DATASPACE\s+SIMPLE\s+\{\s*\(\s*([0-9,\s]+)\s*\)", dump)
    if match is None:
        raise RuntimeError(f"Unable to parse shape for dataset {dataset!r} in {path!r}.")
    return tuple(int(part.strip()) for part in match.group(1).split(",") if part.strip())


def read_h5_dataset_binary(
    path: str,
    dataset: str,
    dtype,
    shape: Sequence[int],
    start: Optional[Sequence[int]] = None,
    count: Optional[Sequence[int]] = None,
):
    cmd = ["h5dump", "-d", dataset]
    if start is not None:
        cmd.extend(["-s", ",".join(str(int(v)) for v in start)])
    if count is not None:
        cmd.extend(["-c", ",".join(str(int(v)) for v in count)])
    cmd.extend(["-b", "LE"])

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
        cmd.extend(["-o", tmp.name, path])
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        arr = np.fromfile(tmp.name, dtype=dtype)

    expected_shape = tuple(int(v) for v in (count if count is not None else shape))
    expected_size = int(np.prod(expected_shape, dtype=np.int64))
    if arr.size != expected_size:
        raise RuntimeError(
            f"Binary dump size mismatch for dataset {dataset!r}: "
            f"expected {expected_size} values, got {arr.size}."
        )
    return arr.reshape(expected_shape)


def load_h5_theta_degrees(path: str, dataset: str = "/exchange/theta") -> np.ndarray:
    dump = _run_h5dump(path, dataset)
    try:
        body = dump.split("DATA {", 1)[1].rsplit("}", 1)[0]
    except IndexError as exc:
        raise RuntimeError(f"Unable to parse h5dump output for {dataset!r} in {path!r}.") from exc

    body = re.sub(r"\(\s*\d+\s*\):", " ", body)
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    values = [float(tok) for tok in re.findall(number_pattern, body)]
    if not values:
        raise RuntimeError(f"No theta values found in dataset {dataset!r} of {path!r}.")
    return np.asarray(values, dtype=np.float64)


def repair_leading_zero_angles_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=np.float64).copy()
    if angles_deg.size == 0:
        raise ValueError("Projection theta array is empty after trimming.")

    nonzero_indices = np.flatnonzero(np.abs(angles_deg) > 1e-12)
    if nonzero_indices.size == 0:
        raise ValueError("Projection theta array contains only zeros.")
    first_valid = int(nonzero_indices[0])
    if first_valid == 0:
        return angles_deg

    diffs = np.diff(angles_deg)
    diffs = diffs[np.abs(diffs) > 1e-12]
    median_step = float(np.median(diffs)) if diffs.size > 0 else float(angles_deg[first_valid] / first_valid)
    if abs(median_step) <= 1e-12:
        raise ValueError("Unable to infer projection angle step.")
    offsets = np.arange(first_valid, 0, -1, dtype=np.float64)
    angles_deg[:first_valid] = angles_deg[first_valid] - median_step * offsets
    print(
        "[RENAF h5] repaired leading zero theta values: "
        f"count={first_valid}, median_step_deg={median_step:.6g}"
    )
    return angles_deg


def centered_crop_offset(src_size, target_size):
    src_size = int(src_size)
    target_size = int(target_size)
    if src_size < target_size:
        raise ValueError(f"Cannot crop source size {src_size} to {target_size}.")
    return (src_size - target_size) // 2


def resolve_projection_alignment(raw_h, raw_w, detector_h, detector_w, event_to_frame_affine):
    linear = np.asarray(event_to_frame_affine, dtype=np.float64)[:, :2]
    scale_x = 1.0 / float(np.linalg.norm(linear[:, 0]))
    scale_y = 1.0 / float(np.linalg.norm(linear[:, 1]))
    crop_src_w = min(int(raw_w), int(np.ceil(float(detector_w) / scale_x)))
    crop_src_h = min(int(raw_h), int(np.ceil(float(detector_h) / scale_y)))
    if crop_src_w < detector_w or crop_src_h < detector_h:
        raise ValueError(
            "Projection alignment crop is smaller than detector: "
            f"crop=[{crop_src_w}, {crop_src_h}], detector=[{detector_w}, {detector_h}]."
        )
    crop_x = centered_crop_offset(raw_w, crop_src_w)
    crop_y = centered_crop_offset(raw_h, crop_src_h)
    return scale_x, scale_y, crop_src_w, crop_src_h, crop_x, crop_y


def scale_and_crop_projection_frames(
    projs_np, detector_h, detector_w, event_to_frame_affine, align_mode="affine"
):
    projs_np = np.asarray(projs_np, dtype=np.float32)
    _, raw_h, raw_w = projs_np.shape

    align_mode = str(align_mode).strip().lower()
    if align_mode in ["center_crop", "crop", "none"]:
        crop_x = centered_crop_offset(raw_w, detector_w)
        crop_y = centered_crop_offset(raw_h, detector_h)
        cropped = projs_np[:, crop_y : crop_y + detector_h, crop_x : crop_x + detector_w]
        return cropped.astype(np.float32, copy=False)
    if align_mode != "affine":
        raise ValueError(
            "raw_data.projection_align_mode must be one of affine or center_crop; "
            f"got {align_mode!r}."
        )

    _, _, crop_w, crop_h, crop_x, crop_y = resolve_projection_alignment(
        raw_h, raw_w, detector_h, detector_w, event_to_frame_affine
    )
    cropped = projs_np[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    if crop_h == detector_h and crop_w == detector_w:
        return cropped.astype(np.float32, copy=False)
    cropped = np.ascontiguousarray(cropped, dtype=np.float32)
    resized = F.interpolate(
        torch.as_tensor(cropped[:, None, :, :]),
        size=(int(detector_h), int(detector_w)),
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0].cpu().numpy().astype(np.float32, copy=False)


def load_projection_subset(h5_path, raw_cfg, projection_indices, detector_h, detector_w):
    data_dataset = raw_cfg.get("data_dataset", "/exchange/data")
    white_dataset = raw_cfg.get("white_dataset", "/exchange/data_white")
    dark_dataset = raw_cfg.get("dark_dataset", "/exchange/data_dark")
    attenuation_eps = float(raw_cfg.get("attenuation_eps", 1e-6))
    apply_minus_log = bool(raw_cfg.get("apply_minus_log", True))
    projection_align_mode = str(raw_cfg.get("projection_align_mode", "affine")).strip().lower()

    data_shape = get_h5_dataset_shape(h5_path, data_dataset)
    white_shape = get_h5_dataset_shape(h5_path, white_dataset)
    dark_shape = get_h5_dataset_shape(h5_path, dark_dataset)
    cache_tag = (
        f"r2_n{len(projection_indices)}_h{detector_h}_w{detector_w}_"
        f"log{int(apply_minus_log)}_align{projection_align_mode}_"
        f"idx{int(projection_indices[0])}_{int(projection_indices[-1])}"
    )
    cache_path = f"{h5_path}.{cache_tag}.projs.npy"
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode=None)
        if cached.ndim == 3:
            return cached.astype(np.float32, copy=False)

    white_cache = f"{h5_path}.white_mean.npy"
    dark_cache = f"{h5_path}.dark_mean.npy"
    if os.path.exists(white_cache):
        white_mean = np.load(white_cache, mmap_mode=None)
    else:
        white = read_h5_dataset_binary(h5_path, white_dataset, np.uint16, white_shape).astype(np.float32)
        white_mean = np.mean(white, axis=0, dtype=np.float32)
        np.save(white_cache, white_mean)

    if os.path.exists(dark_cache):
        dark_mean = np.load(dark_cache, mmap_mode=None)
    else:
        dark = read_h5_dataset_binary(h5_path, dark_dataset, np.uint16, dark_shape).astype(np.float32)
        dark_mean = np.mean(dark, axis=0, dtype=np.float32)
        np.save(dark_cache, dark_mean)

    denom = np.maximum(white_mean - dark_mean, attenuation_eps)
    frame_h, frame_w = int(data_shape[1]), int(data_shape[2])
    frames = []
    for idx in projection_indices:
        frame = read_h5_dataset_binary(
            h5_path,
            data_dataset,
            np.uint16,
            data_shape,
            start=(int(idx), 0, 0),
            count=(1, frame_h, frame_w),
        )[0].astype(np.float32)
        frame = np.maximum(frame - dark_mean, attenuation_eps) / denom
        frame = np.clip(frame, attenuation_eps, None)
        if apply_minus_log:
            frame = -np.log(frame)
        frames.append(frame.astype(np.float32, copy=False))

    frames = np.stack(frames, axis=0)
    event_to_frame_affine = raw_cfg.get(
        "event_to_frame_affine",
        [[1.0927234, -0.00763725, -63.842716], [0.00763725, 1.0920073, -3.5674274]],
    )
    frames = scale_and_crop_projection_frames(
        frames, detector_h, detector_w, event_to_frame_affine, projection_align_mode
    )
    try:
        np.save(cache_path, frames)
    except Exception:
        pass
    return frames


def load_event_csv(path: str, column_order: Iterable[str] = ("x", "y", "p", "t"), use_cache: bool = True):
    cache_path = f"{path}.npy"
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode=None)
        if cached.dtype == RAW_EVENT_DTYPE:
            return cached

    raw = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if raw.ndim == 1:
        raw = raw[None, :]
    order = [str(name).strip().lower() for name in column_order]
    if sorted(order) != ["p", "t", "x", "y"]:
        raise ValueError(f"Unsupported event CSV column order {column_order!r}.")
    col = {name: idx for idx, name in enumerate(order)}
    events = np.empty(raw.shape[0], dtype=RAW_EVENT_DTYPE)
    events["x"] = raw[:, col["x"]].astype(np.int32, copy=False)
    events["y"] = raw[:, col["y"]].astype(np.int32, copy=False)
    events["p"] = raw[:, col["p"]].astype(np.int16, copy=False)
    events["t"] = raw[:, col["t"]].astype(np.int64, copy=False)
    events["p"][events["p"] == 0] = -1
    events.sort(order="t")
    if use_cache:
        try:
            np.save(cache_path, events)
        except Exception:
            pass
    return events
