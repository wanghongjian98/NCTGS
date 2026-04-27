from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert directories of 2D projection .npy files into mp4 videos."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Root directory containing subdirectories of .npy frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where mp4 videos will be written.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="Upper percentile used for intensity clipping before uint8 conversion.",
    )
    return parser.parse_args()


def find_sequences(input_root: Path) -> list[Path]:
    return sorted(
        path for path in input_root.iterdir() if path.is_dir() and any(path.glob("*.npy"))
    )


def compute_scale(files: list[Path], clip_percentile: float) -> tuple[float, float]:
    mins = []
    maxs = []
    samples = []
    sample_step = max(1, len(files) // 32)

    for idx, path in enumerate(files):
        frame = np.load(path)
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame in {path}, got shape={frame.shape}")
        mins.append(float(frame.min()))
        maxs.append(float(frame.max()))
        if idx % sample_step == 0:
            samples.append(frame.reshape(-1))

    sample_values = np.concatenate(samples) if samples else np.array([0.0], dtype=np.float32)
    low = min(mins)
    high = float(np.percentile(sample_values, clip_percentile))
    high = max(high, max(maxs) * 0.1, low + 1e-6)
    return low, high


def to_uint8(frame: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(frame, low, high)
    scaled = (clipped - low) / (high - low)
    gray = np.round(scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def encode_sequence(files: list[Path], output_path: Path, fps: int, low: float, high: float) -> None:
    first = np.load(files[0])
    if first.ndim != 2:
        raise ValueError(f"Expected 2D frame in {files[0]}, got shape={first.shape}")
    height, width = first.shape

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert process.stdin is not None

    try:
        for path in files:
            frame = np.load(path)
            if frame.shape != (height, width):
                raise ValueError(
                    f"Inconsistent frame shape in {path}: expected {(height, width)}, got {frame.shape}"
                )
            process.stdin.write(to_uint8(frame, low, high).tobytes())
    finally:
        process.stdin.close()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sequences = find_sequences(args.input_root)
    if not sequences:
        raise FileNotFoundError(f"No subdirectories with .npy files found in {args.input_root}")

    for sequence_dir in sequences:
        files = sorted(sequence_dir.glob("*.npy"))
        low, high = compute_scale(files, args.clip_percentile)
        output_path = args.output_dir / f"{sequence_dir.name}.mp4"
        print(
            f"Encoding {sequence_dir.name}: frames={len(files)}, scale=[{low:.6f}, {high:.6f}] -> {output_path}"
        )
        encode_sequence(files, output_path, args.fps, low, high)


if __name__ == "__main__":
    main()
