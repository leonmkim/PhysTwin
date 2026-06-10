# PhysTwin UV Install Notes

Isolated `uv` environment for `third_party/phystwin`. All `uv` commands run from this
directory only — do not sync from the `belt_perception` root.

## Validated environment (Stage A/B, 2026-06-09)

| Component | Value |
|---|---|
| Python | 3.10 (pinned via `.python-version`) |
| nvcc | CUDA 12.6 (`/usr/local/cuda-12.6`) |
| `CUDA_HOME` | `/usr/local/cuda-12.6` |
| PyTorch | `2.4.0+cu121` (`torch.version.cuda` reports 12.1) |
| warp-lang | `1.14.0` |
| GPU | NVIDIA GeForce RTX 4090 (`sm_89`) |
| Native extensions | `diff-gaussian-rasterization`, `simple-knn` built successfully under nvcc 12.6 + cu121 wheels |

CUDA 12.1 toolkit is a **fallback only** if later native extension or runtime failures
appear. Do not install or require CUDA 12.1 unless that fallback is needed.

## Install (dependency groups)

```bash
cd third_party/phystwin
uv python pin 3.10

# Stage A
uv sync --group core

# Stage B (playground + train + gaussian native extensions + pytorch3d wheel)
uv sync --group core --group playground --group gaussian --group train
```

`pytorch3d==0.7.8` is in the `gaussian` group and locked via a direct wheel URL in
`pyproject.toml` (`py310_cu121_pyt240`). A fresh `uv sync` installs it without a
separate pip step.

### PyTorch3D fallback (only if direct wheel lock/sync fails)

```bash
uv pip install --no-index \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html \
  pytorch3d==0.7.8
```

## Runtime environment (local validation)

Run `uv sync` **before** copying `.envrc.example` → `.envrc` (or re-run `direnv allow`
after sync). If direnv loads while `.venv` is empty or torch is not yet installed,
`LD_LIBRARY_PATH` will not include PyTorch's `lib/` and CUDA extension imports will fail
with errors like `libc10.so: cannot open shared object file`.

```bash
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.9+PTX"   # RTX 4090; adjust for your GPU

# Required so CUDA extensions (simple-knn, diff-gaussian-rasterization) find libc10
export LD_LIBRARY_PATH="$(
  uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'
):${LD_LIBRARY_PATH:-}"
```

If using direnv: after `uv sync`, run `direnv reload` (or re-enter the directory) so
`.envrc` re-resolves the torch lib path from the populated `.venv`.

Headless import/run paths (e.g. `pynput`) need a display. Use `xvfb-run` or a local X
server:

```bash
mkdir -p temp_experiments/logs
timeout 180s xvfb-run -a \
  uv run python interactive_playground.py \
    --case_name double_stretch_sloth \
    --n_ctrl_parts 2 \
  2>&1 | tee temp_experiments/logs/interactive_playground_double_stretch_sloth.log
```

## Playground smoke test (validated 2026-06-09)

| Item | Result |
|---|---|
| Case | `double_stretch_sloth` |
| Command | `timeout 180s xvfb-run -a uv run python interactive_playground.py --case_name double_stretch_sloth --n_ctrl_parts 2` (with runtime env above) |
| Exit code | `124` from `timeout` — **acceptable** (infinite render loop; process was healthy) |
| Artifacts loaded | `experiments_optimization/double_stretch_sloth/optimal_params.pkl`, `data/different_types/double_stretch_sloth/final_data.pkl`, `experiments/double_stretch_sloth/train/best_199.pth` |
| Render loop | ~35–36 FPS sustained (simulator + Gaussian rendering + compositing) |
| Errors | No traceback, CUDA symbol errors, missing-artifact asserts, or segfaults |
| Log | `temp_experiments/logs/interactive_playground_double_stretch_sloth.log` (gitignored; do not commit) |

## Stage C scratch output roots (output-safety)

Training, optimization, inference, and Gaussian render scripts accept explicit
output-root flags. **Upstream defaults are unchanged** when flags are omitted.

| Scratch write root | CLI / env override |
|---|---|
| `temp_experiments_optimization_uv/` | `--experiments-optimization-dir` |
| `temp_experiments_uv/` | `--experiments-dir` |
| `temp_gaussian_output_uv/` | `--gaussian-output-dir` or `GAUSSIAN_OUTPUT_DIR` |
| `temp_gaussian_output_dynamic_uv/` | `--gaussian-output-dynamic-dir` or `GAUSSIAN_OUTPUT_DYNAMIC_DIR` |
| `temp_gaussian_output_dynamic_white_uv/` | `GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR` |
| `temp_gaussian_data_uv/` | `INTERP_POSES_OUTPUT_DIR` (with `GAUSSIAN_DATA_DIR` aligned; see below) |

**`gs_run.sh` gaussian data coupling:** `generate_interp_poses.py` reads camera
metadata from `GAUSSIAN_DATA_DIR` (default `./data/gaussian_data`). If
`INTERP_POSES_OUTPUT_DIR` points at a scratch root, set `GAUSSIAN_DATA_DIR` to that
same scratch root (after copying or symlinking scene dirs from reference data), or
poses will not be visible to `gs_train.py -s`. Never write `interp_poses.pkl` into
author `data/gaussian_data` when using scratch mode.

**`gs_run_simulate.sh` / `gs_run_simulate_white.sh` gaussian data coupling:**
`gs_render_dynamics.py` loads the scene from `-s` (wired via `gaussian_data_dir`,
default `./data/gaussian_data/<scene>`). Dataset init may rewrite `points3D.ply`
under that path. For scratch warp + scratch Gaussian dynamic renders, set
`GAUSSIAN_DATA_DIR=./temp_gaussian_data_uv` together with `REFERENCE_EXPERIMENTS_DIR`,
`REFERENCE_GAUSSIAN_OUTPUT_DIR`, and `GAUSSIAN_OUTPUT_DYNAMIC_DIR` (or
`GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR` for the white-background script).

Read author reference artifacts while writing scratch outputs:

| Reference read | Flag / env |
|---|---|
| Author `experiments_optimization/` | `--reference-experiments-optimization-dir experiments_optimization` |
| Author `experiments/` (checkpoints, inference.pkl) | `--reference-experiments-dir experiments` |
| Author `gaussian_output/` (trained GS models) | `--reference-gaussian-output-dir ./gaussian_output` or `REFERENCE_GAUSSIAN_OUTPUT_DIR` |

Example (optimize → train using author optimal params, scratch writes only):

```bash
cd third_party/phystwin
mkdir -p temp_experiments_optimization_uv temp_experiments_uv

uv run python optimize_cma.py \
  --base_path ./data/different_types \
  --case_name double_stretch_sloth \
  --train_frame <frame> \
  --experiments-optimization-dir temp_experiments_optimization_uv

export WANDB_MODE=disabled

uv run python train_warp.py \
  --base_path ./data/different_types \
  --case_name double_stretch_sloth \
  --train_frame <frame> \
  --iterations 1 \
  --disable-video-logging \
  --experiments-optimization-dir temp_experiments_optimization_uv \
  --reference-experiments-optimization-dir experiments_optimization \
  --experiments-dir temp_experiments_uv
```

Use `--iterations 1` or `--iterations 2` with `--disable-video-logging` for tiny
smoke runs when headless ffmpeg cannot encode `avc1` (default config iteration
count is much larger). Set `WANDB_MODE=disabled` to avoid wandb overhead.

Do **not** run full training/render pipelines until this patch is reviewed. Use
`--help` on entry points and `bash -n` on shell drivers to validate parsers only.

## Stage E preprocessing (raw RGB-D → processed artifacts)

Stage E validates the README “Data Processing from Raw Videos” path. **Stage E0**
(output-safety + dependency scaffolding) does not run preprocessing or install
heavy models.

### Raw case input layout

Each case directory under the processed root contains at minimum:

- `color/` — per-camera `{i}.mp4` and `color/{i}/{frame}.png`
- `depth/` — per-camera `{i}/{frame}.npy` (depth in mm)
- `calibrate.pkl` — camera extrinsics
- `metadata.json` — `intrinsics`, `WH`, `frame_num`, etc.

Downstream steps add `mask/`, `cotracker/`, `pcd/`, `shape/`, `final_data.pkl`,
`split.json`, and export targets under `gaussian_data/`.

### Scratch roots (do not write author `data/`)

| Scratch root | Purpose |
|---|---|
| `temp_raw_data_uv/<case>/` | Seeded raw RGB-D copy (read author `data/` only) |
| `temp_processed_data_uv/<case>/` | `process_data.py` outputs |
| `temp_gaussian_data_uv/<case>/` | `export_gaussian_data.py` outputs |
| `temp_masks_uv/<case>/` | `export_video_human_mask.py` outputs |
| `temp_preprocess_results_uv/` | Logs, compare reports, optional `--timer-log` |

**Never** use symlinked author `data/different_types` as `--base-path` when running
exports or `process_data.py` in scratch-validation mode. `export_video_human_mask.py`
removes `{base_path}/{case}/tmp_data` after each camera; that side effect is
confined to the selected `--base-path`.

### Output-safety flags (defaults match upstream)

| Script | Flags | Default |
|---|---|---|
| `process_data.py` | `--base_path`, `--timer-log` | `./data/different_types`, `timer.log` |
| `script_process_data.py` | `--base-path`, `--timer-log` | `./data/different_types`, `timer.log` |
| `export_gaussian_data.py` | `--base-path`, `--output-path`, `--case-name` | `./data/different_types`, `./data/gaussian_data` |
| `export_video_human_mask.py` | `--base-path`, `--output-path`, `--case-name` | `./data/different_types`, `./data/different_types_human_mask` |
| `export_render_eval_data.py` | `--base-path`, `--output-path`, `--case-name` | `./data/different_types`, `./data/render_eval_data` |

`script_process_data.py` removes `--timer-log` at start (upstream removed `timer.log`
in cwd). Point `--timer-log` at `temp_preprocess_results_uv/timer.log` for scratch runs.

### Safe single-case examples (after Stage E1+ deps installed)

```bash
cd third_party/phystwin
mkdir -p temp_preprocess_results_uv/logs

# Seed raw (read-only copy from author reference)
mkdir -p temp_raw_data_uv
rsync -a \
  --include='color/***' --include='depth/***' \
  --include='calibrate.pkl' --include='metadata.json' \
  --exclude='*' \
  data/different_types/double_stretch_sloth/ \
  temp_raw_data_uv/double_stretch_sloth/

cp -a temp_raw_data_uv/double_stretch_sloth temp_processed_data_uv/

xvfb-run -a uv run python process_data.py \
  --base_path ./temp_processed_data_uv \
  --case_name double_stretch_sloth \
  --category sloth \
  --shape_prior \
  --timer-log temp_preprocess_results_uv/timer.log

uv run python export_gaussian_data.py \
  --base-path ./temp_processed_data_uv \
  --output-path ./temp_gaussian_data_uv \
  --case-name double_stretch_sloth
```

Do **not** run the above until segmentation/TRELLIS/SDXL/CoTracker deps are installed
(Stage E1+). Use `--help` only during Stage E0.

### Dependency groups (submodule `pyproject.toml` only)

```bash
cd third_party/phystwin

# Lightweight scaffolding (Stage E0/E1)
uv sync --group core --group playground --group preprocessing-core

# Optional PyPI groups (install only when needed)
uv sync --group preprocessing-diffusion
uv sync --group preprocessing-realsense   # collection / custom data only
```

**Manual / git-heavy (documented, not uv-locked):**

- Grounded-SAM-2 + GroundingDINO — shared clone + per-worktree editable install
  (see **Stage E2d segmentation environment** below)
- Checkpoints — shared cache under `PHYSTWIN_SEG_CHECKPOINT_DIR` (download requires
  explicit approval; do not duplicate per worktree)
- CoTracker — `torch.hub.load("facebookresearch/co-tracker", ...)` at runtime
- TRELLIS — clone under `data_process/TRELLIS`, run `setup.sh` (native stack)

### Stage E2d segmentation environment (shared cache)

Segmentation scripts hardcode paths relative to the PhysTwin repo root:

- SAM2 config: `configs/sam2.1/sam2.1_hiera_l.yaml`
- Checkpoints: `data_process/groundedSAM_checkpoints/*.pt` / `*.pth`
- `groundingdino` imports require `PYTHONPATH` to include the Grounded-SAM-2 clone root

Use a **shared persistent cache** across worktrees (not per-worktree clones):

```bash
export PHYSTWIN_EXTERNAL_ROOT=/mnt/data2/magna/belt_perception/third_party/phystwin_external
export GSAM_ROOT="${PHYSTWIN_EXTERNAL_ROOT}/Grounded-SAM-2"
export PHYSTWIN_SEG_CHECKPOINT_DIR="${PHYSTWIN_EXTERNAL_ROOT}/checkpoints"
```

**One-time shared clone** (network/git; requires explicit approval):

```bash
mkdir -p "${PHYSTWIN_EXTERNAL_ROOT}"
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "${GSAM_ROOT}"
```

**Per-worktree venv install** (after preprocessing-core sync):

```bash
cd third_party/phystwin
uv sync --group core --group playground --group train --group preprocessing-core
uv pip install -e "${GSAM_ROOT}"
uv pip install --no-build-isolation -e "${GSAM_ROOT}/grounding_dino"
uv pip install 'transformers<5' yapf timm addict pycocotools
```

**Shared checkpoint placement** (download/copy once; not committed):

```text
${PHYSTWIN_SEG_CHECKPOINT_DIR}/sam2.1_hiera_large.pt
${PHYSTWIN_SEG_CHECKPOINT_DIR}/groundingdino_swint_ogc.pth
```

Checkpoint download requires explicit approval. When approved, download into the
shared checkpoint dir (not into each worktree). `env_install/download_pretrained_models.sh`
writes under `data_process/groundedSAM_checkpoints/` by default — move/copy the `.pt` /
`.pth` files into `${PHYSTWIN_SEG_CHECKPOINT_DIR}/` afterward.

**Per-worktree activation** (symlinks + `PYTHONPATH`; no install/download):

```bash
cd third_party/phystwin
source scripts/setup_segmentation_env.sh
```

Scratch fallback for a single worktree only:

```bash
export PHYSTWIN_EXTERNAL_ROOT="$(pwd)/temp_external_uv"
source scripts/setup_segmentation_env.sh
```

**First approved segmentation smoke (not run in E2d hygiene):** camera 0 only, scratch
outputs under `temp_processed_data_uv/`, from PhysTwin repo root:

```bash
cd third_party/phystwin
source scripts/setup_segmentation_env.sh

env -u PYTHONPATH PYTHONPATH="${GSAM_ROOT}" \
  timeout 7200s xvfb-run -a uv run python data_process/segment_util_video.py \
    --base_path ./temp_processed_data_uv \
    --case_name double_stretch_sloth \
    --TEXT_PROMPT "sloth.hand" \
    --camera_idx 0
```

Do not write author `data/`. Use `--base_path ./temp_processed_data_uv` only.

**Environment note:** README warns that TRELLIS conflicts with
`diff-gaussian-rasterization`. Prefer a preprocessing-focused sync without
`--group gaussian` until preprocessing is validated; add `gaussian` only for
Stage D-style training/render work.

### Stage E3 shape-prior environment (shared cache)

Shape-prior substages (`image_upscale.py`, `segment_util_image.py`, `shape_prior.py`,
`align.py`, `data_process_sample.py --shape_prior`) were **validated in scratch** for
`double_stretch_sloth` (E3d–E3i, 2026-06-10). Stage E3b covers helper setup and
dependency pins; see parent-repo report
`docs/experiments/phystwin_stage_e_shape_prior_preprocessing_smoke_double_stretch_sloth.md`
for per-stage commands, timings, and comparisons.

**Not validated yet:** full `process_data.py` orchestration; `export_gaussian_data.py`
from scratch `final_data.pkl`.

**Shared cache layout** (extends segmentation cache):

```text
${PHYSTWIN_EXTERNAL_ROOT}/
├── Grounded-SAM-2/           # segmentation (E2d)
├── checkpoints/              # SAM2 + GroundingDINO (E2d)
├── TRELLIS/                  # one shared git clone
├── huggingface/              # HF_HOME / HUGGINGFACE_HUB_CACHE
│   └── hub/
├── superglue_weights/        # SuperPoint + SuperGlue .pth files
└── torch_hub/                # optional TORCH_HOME
```

Default `PHYSTWIN_EXTERNAL_ROOT`:

```bash
/mnt/data2/magna/belt_perception/third_party/phystwin_external
```

**One-time TRELLIS clone** (network/git; requires explicit approval):

```bash
export PHYSTWIN_EXTERNAL_ROOT=/mnt/data2/magna/belt_perception/third_party/phystwin_external
mkdir -p "${PHYSTWIN_EXTERNAL_ROOT}"
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git \
  "${PHYSTWIN_EXTERNAL_ROOT}/TRELLIS"
```

**TRELLIS `setup.sh` vs uv venv:** upstream `setup.sh --basic` calls `pip` and may
install unpinned `transformers`. In uv worktrees, prefer explicit `uv pip install`
(below). Preserve PhysTwin pins: `transformers<5`, `diffusers>=0.27,<0.31`.

**Avoid `uv sync` immediately before shape-prior smoke** unless revalidating native
deps — sync can remove manually installed TRELLIS packages (E3f–E3g).

**Per-worktree uv sync** (SDXL / HF stack baseline):

```bash
cd third_party/phystwin
uv sync --group core --group playground --group train \
  --group preprocessing-core --group preprocessing-diffusion
```

`preprocessing-diffusion` pins `transformers<5` (GroundingDINO) and
`diffusers>=0.27,<0.31` (torch 2.4 / SDXL). After sync, reinstall per-worktree GSAM
editable packages (Stage E2d) and the native stack below.

**OpenCV repair (E3d):** if `cv2.__file__` is `None` after sync:

```bash
uv pip install numpy==1.26.4 opencv-python==4.11.0.86
```

#### Validated TRELLIS native stack (E3f–E3g)

Minimum for `TrellisImageTo3DPipeline` import + `from_pretrained` + `.cuda()`:

```bash
cd third_party/phystwin
# TRELLIS setup.sh --basic equivalent (do not run bare setup.sh in uv venv)
uv pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
  scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph
uv pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
uv pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
uv pip install spconv-cu120==2.3.6
```

Runtime env (match `shape_prior.py` + E3f workaround):

```bash
export ATTN_BACKEND=xformers    # required unless flash-attn is installed
export SPCONV_ALGO=native       # shape_prior.py sets this in-script
```

Import smoke:

```bash
source scripts/setup_shape_prior_env.sh
env -u PYTHONPATH PYTHONPATH="${TRELLIS_ROOT}:${PWD}/data_process" ATTN_BACKEND=xformers \
  uv run --no-sync python -c \
  "from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline; print('TRELLIS import OK')"
```

**Additional for full `shape_prior.py` generation** (E3g):

```bash
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
uv pip install --no-build-isolation /tmp/extensions/nvdiffrast

git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
uv pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```

`flash-attn` was **not** installed in validation; `ATTN_BACKEND=xformers` was used
instead of `shape_prior.py`'s default `flash_attn`.

**HF / Torch caches (approved downloads, shared):**

| Asset | Cache path |
|---|---|
| SDXL upscaler | `${HF_HOME}/hub/models--stabilityai--stable-diffusion-x4-upscaler` |
| GroundingDINO BERT | `${HF_HOME}/hub/models--bert-base-uncased` |
| TRELLIS weights | `${HF_HOME}/hub/models--JeffreyXiang--TRELLIS-image-large` (~2.9 GB) |
| DINOv2 cond model | `${TORCH_HOME}/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth` (~1.13 GB) |

**PyTorch3D wheel only** (align render path; avoid full `--group gaussian`):

```bash
cd third_party/phystwin
uv pip install \
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
env -u PYTHONPATH uv run --no-sync python -c \
  "from pytorch3d.renderer import look_at_view_transform; print('pytorch3d OK')"
```

**SuperGlue shared weights** (download once; not committed):

```text
${PHYSTWIN_EXTERNAL_ROOT}/superglue_weights/superpoint_v1.pth
${PHYSTWIN_EXTERNAL_ROOT}/superglue_weights/superglue_indoor.pth
${PHYSTWIN_EXTERNAL_ROOT}/superglue_weights/superglue_outdoor.pth
```

When approved, download into the shared dir (see `env_install/download_pretrained_models.sh`
for URLs). Per-worktree symlinks are created by the helper below.

**Per-worktree activation** (symlinks + cache env; no download/generation):

```bash
cd third_party/phystwin
source scripts/setup_shape_prior_env.sh
```

For segmentation stages in the same run, also source segmentation setup:

```bash
source scripts/setup_segmentation_env.sh
source scripts/setup_shape_prior_env.sh
```

**Validated substage commands (scratch only; `double_stretch_sloth`):**

```bash
cd third_party/phystwin
source scripts/setup_segmentation_env.sh
source scripts/setup_shape_prior_env.sh
CASE=temp_processed_data_allcam_uv/double_stretch_sloth

# E3d SDXL upscale
uv run python ./data_process/image_upscale.py \
  --img_path ./${CASE}/color/0/0.png \
  --mask_path ./${CASE}/mask/0/0/0.png \
  --output_path ./${CASE}/shape/high_resolution.png --category sloth

# E3e image segmentation (note: --img_path/--output_path, not --base_path)
env -u PYTHONPATH PYTHONPATH="${GSAM_ROOT}:${TRELLIS_ROOT}:${PWD}/data_process" \
  xvfb-run -a uv run python ./data_process/segment_util_image.py \
  --img_path ./${CASE}/shape/high_resolution.png --TEXT_PROMPT sloth \
  --output_path ./${CASE}/shape/masked_image.png

# E3g TRELLIS generation
env -u PYTHONPATH PYTHONPATH="${TRELLIS_ROOT}:${PWD}/data_process" ATTN_BACKEND=xformers \
  xvfb-run -a uv run --no-sync python ./data_process/shape_prior.py \
  --img_path ./${CASE}/shape/masked_image.png --output_dir ./${CASE}/shape

# E3h align (remove stale matching/ first if author mesh was copied earlier)
rm -rf ./${CASE}/shape/matching
env -u PYTHONPATH PYTHONPATH="${TRELLIS_ROOT}:${PWD}/data_process" \
  xvfb-run -a uv run --no-sync python ./data_process/align.py \
  --base_path ./temp_processed_data_allcam_uv --case_name double_stretch_sloth \
  --controller_name hand

# E3i final sample
cd data_process
env -u PYTHONPATH PYTHONPATH="${TRELLIS_ROOT}:${PWD}" \
  xvfb-run -a uv run --no-sync python data_process_sample.py \
  --base_path ../temp_processed_data_allcam_uv --case_name double_stretch_sloth --shape_prior
```

**Headless MP4 caveat:** OpenCV `VideoWriter` with `avc1`/H.264 may fail in xvfb-only
environments. Missing `visualization.mp4` / `final_matching.mp4` / `final_data.mp4` is
non-fatal when primary GLB/PKL artifacts exist.

Do not write author `data/`. Route outputs under `temp_*_uv/{case_name}/` only.

## Author artifacts (not committed)

Large datasets live on shared storage. Symlink into this directory (see canonical
checkout or `.envrc.example`). Expected paths for playground validation:

- `data/different_types/<case_name>/`
- `experiments_optimization/<case_name>/optimal_params.pkl`
- `experiments/<case_name>/train/best_*.pth`
- `gaussian_output/<case_name>/.../point_cloud/iteration_10000/point_cloud.ply`

## Import sanity checks

```bash
uv run python -c "import numpy, torch, warp as wp; wp.init(); print('core OK')"
uv run python -c "import diff_gaussian_rasterization, simple_knn; print('native ext OK')"
uv run python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"
xvfb-run -a uv run python -c "from qqtt import InvPhyTrainerWarp; import gs_render; print('closure OK')"
```

## CUDA 12.1 fallback (only if needed)

If native extension builds fail under nvcc 12.6 despite correct `TORCH_CUDA_ARCH_LIST`:

1. Load or install CUDA 12.1 toolkit and set `CUDA_HOME` to that path.
2. Rebuild: `uv sync --group gaussian --reinstall-package diff-gaussian-rasterization --reinstall-package simple-knn`
3. Re-run import checks above.

Do not downgrade PyTorch wheels unless PyTorch3D / extension compatibility requires it.
