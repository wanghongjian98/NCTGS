# NCTGS: Event-Guided Lattice CT Gaussian Splatting

This repository is our working NCTGS project built from
[R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian). The current codebase
keeps the original R2-Gaussian xray rasterization and voxelization kernels, and
adds the RENAF raw h5 pipeline, full-resolution projection training, original
volume reconstruction, selected-slice evaluation, and event-guided lattice
training.

The main target dataset in this checkout is the RENAF `test16` scan:

- projection input: `477 x 720 x 1280`
- detector resolution: `1280 x 720`
- reconstruction volume: `1280 x 1280 x 720`
- initialization: `configs/init_renaf_test16_source_proj_medium_eval_center_crop_75k.npy`

## What Changed

The project extends R2-Gaussian in four places.

1. Raw RENAF h5 loading
   - Implemented in `r2_gaussian/dataset/raw_h5_reader.py`.
   - Reads h5 datasets through `h5dump`, so the training path does not depend on
     `h5py`.
   - Loads `/exchange/theta`, repairs leading zero theta values, applies
     white/dark correction, converts projections with `-log`, and supports
     detector alignment through center crop or affine transforms.
   - Caches projection arrays and white/dark averages next to the h5 data.

2. RENAF scanner and volume setup
   - Implemented in `r2_gaussian/dataset/dataset_readers.py`.
   - Converts RENAF YAML configs to the R2-Gaussian scanner model.
   - Builds or loads cached FBP ground-truth volumes for evaluation.
   - Supports projection-only, event-only, and projection-plus-event training
     modes.

3. Event-guided lattice training
   - Implemented in `train_e.py`.
   - Samples event pairs from the RENAF event stream, maps timestamps to angular
     bins, renders reference/current projections, and applies a polarity-aware
     softplus event loss.
   - The event branch is controlled by `configs/renaf_test16_h5_event.yaml`.

4. Practical full-volume evaluation
   - Implemented in `train.py`.
   - Full 3D evaluation at `1280 x 1280 x 720` is expensive, so the default
     config evaluates 2D projections every 1000 iterations and only queries a
     small set of selected Z slices every 1000 iterations.
   - Full 3D volume metrics are deferred to `eval_3d_interval`, currently the
     final iteration.

## Active Backend

The active rendering path is the original R2-Gaussian CUDA extension:

```text
r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

Earlier experiments with `gs-voxelizer` and FaCT-GS-style kernels are not active
in the current code or config. This avoids mixing kernel assumptions while we are
running the RENAF full-resolution setup.

## Installation

Use the CUDA version that matches the PyTorch build. The current environment was
set up for PyTorch CUDA 11.8, so load CUDA 11.8 before building extensions.

```bash
cd /das/work/p20/p20846/RefractoryEventCT/eventgs/r2_gaussian
conda activate /das/work/p20/p20846/wang_hongjian/environment/envs/r2_gaussian

module purge
module load cuda/11.8.0

pip install -r requirements.txt
pip install -e r2_gaussian/submodules/simple-knn
pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

TIGRE is required for FBP initialization and volume evaluation:

```bash
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation
```

If extension compilation reports a CUDA mismatch, check:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
PY
nvcc --version
```

The CUDA reported by `nvcc` should match `torch.version.cuda`.

## Projection Training

The main full-resolution projection config is:

```text
configs/renaf_test16_h5_proj_medium_eval.yaml
```

Run it from the repository root:

```bash
cd /das/work/p20/p20846/RefractoryEventCT/eventgs/r2_gaussian
conda activate /das/work/p20/p20846/wang_hongjian/environment/envs/r2_gaussian
module purge
module load cuda/11.8.0

python train.py --config configs/renaf_test16_h5_proj_medium_eval.yaml
```

The source geometry for that run is:

```text
configs/renaf_test16_source_proj_medium_eval.yaml
```

Current key settings:

```yaml
geometry:
  mode: parallel
  nDetector: [1280, 720]
  nVoxel: [1280, 1280, 720]
train:
  mode: proj
  proj_num: 477
```

Training output is written to:

```text
output/renaf_test16_h5_proj_medium_eval
```

Useful outputs:

- `cfg_args.yml`: runtime snapshot of the config used by the current run
- `point_cloud/iteration_*/point_cloud.ply`: Gaussian checkpoints
- `chkpnt*.pth`: optimizer checkpoints
- `eval/iter_*/eval2d_render_*.png`: projection render previews
- `eval/iter_*/eval3d_selected_slices.png`: selected Z-slice volume previews
- `eval/iter_*/eval3d.yml`: full 3D metrics when full 3D eval is scheduled

`cfg_args.yml` is produced when training starts. If a config file is changed
while training is already running, restart training to refresh the runtime
snapshot.

## Event-Guided Lattice Training

Use the event config when training with RENAF events:

```bash
python train_e.py --config configs/renaf_test16_h5_event.yaml
```

The event implementation uses:

- `event_lambda`: event loss weight
- `event_pairs_per_iter`: number of event pairs sampled per iteration
- `event_tau_us`: event time window in microseconds
- `event_angle_bins`: angular bins used to map event timestamps to render views

The event loss is implemented in `compute_event_loss` in `train_e.py`. It samples
events, renders the two corresponding lattice states, and penalizes projection
changes that disagree with the event polarity.

## Initialization

Gaussian initialization is handled by:

```text
r2_gaussian/gaussian/initialize.py
```

For the current full-volume RENAF run, the initialization file is:

```text
configs/init_renaf_test16_source_proj_medium_eval_center_crop_75k.npy
```

It contains 75k initial points in normalized R2-Gaussian coordinates and is
compatible with the original `1280 x 1280 x 720` volume setup.

## Evaluation Policy

Full 3D eval at `1280 x 1280 x 720` is very expensive. The current config is set
up to make progress visible without running full-volume eval every 1000 steps:

```yaml
test_iterations: [1000, 2000, ..., 30000]
eval_max_cameras: 12
eval_3d_interval: 30000
eval_slice_interval: 1000
eval_slice_count: 5
eval_save_projection_png: true
eval_projection_png_count: 5
```

This means:

- every 1000 iterations: evaluate a small set of 2D projections
- every 1000 iterations: save selected volume slices only
- final iteration: run the full 3D volume query and metrics

## Notes

- Projection resolution and volume resolution are independent. The current
  projection tensor is `720 x 1280`; the reconstruction volume is
  `1280 x 1280 x 720`.
- `lambda_tv: 0.0` disables TV loss in the current fast projection config.
- `views_per_iter: 1` keeps each optimization step lighter. Increase it only if
  GPU time and memory allow it.
- The code keeps R2-Gaussian kernel semantics. Do not install incompatible CUDA
  extensions into the same environment unless the training path is explicitly
  changed to use them.

## Acknowledgement

This project is based on R2-Gaussian:

```bibtex
@inproceedings{zha2024r2gaussian,
  title={R2-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
  author={Zha, Ruyi and Zhang, Tao and Li, Hongdong},
  booktitle={NeurIPS},
  year={2024}
}
```
