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

# Stage B (playground + train + gaussian native extensions)
uv sync --group core --group playground --group gaussian --group train
```

## PyTorch3D (post-sync, not in uv.lock)

Install after `uv sync` using upstream find-links:

```bash
uv pip install --no-index \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html \
  pytorch3d==0.7.8
```

## Runtime environment (local validation)

```bash
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.9+PTX"   # RTX 4090; adjust for your GPU

# Required so CUDA extensions (simple-knn, diff-gaussian-rasterization) find libc10
export LD_LIBRARY_PATH="$(
  uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'
):${LD_LIBRARY_PATH:-}"
```

Headless import/run paths (e.g. `pynput`) need a display. Use `xvfb-run` or a local X
server:

```bash
xvfb-run -a uv run python interactive_playground.py \
  --case_name double_stretch_sloth --n_ctrl_parts 2
```

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
