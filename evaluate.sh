#!/usr/bin/env bash
# Output-root overrides (Stage C). Defaults preserve upstream paths.
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-experiments}"
REFERENCE_EXPERIMENTS_DIR="${REFERENCE_EXPERIMENTS_DIR:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
GAUSSIAN_OUTPUT_DYNAMIC_DIR="${GAUSSIAN_OUTPUT_DYNAMIC_DIR:-./gaussian_output_dynamic}"
GAUSSIAN_DATA_DIR="${GAUSSIAN_DATA_DIR:-./data/gaussian_data}"
RENDER_EVAL_DATA_DIR="${RENDER_EVAL_DATA_DIR:-./data/render_eval_data}"
HUMAN_MASK_PATH="${HUMAN_MASK_PATH:-./data/different_types_human_mask}"

common_args=(
    --experiments-dir "${EXPERIMENTS_DIR}"
    --results-dir "${RESULTS_DIR}"
)
if [[ -n "${REFERENCE_EXPERIMENTS_DIR}" ]]; then
    common_args+=(--reference-experiments-dir "${REFERENCE_EXPERIMENTS_DIR}")
fi

python evaluate_chamfer.py "${common_args[@]}"
python evaluate_track.py "${common_args[@]}"
python gaussian_splatting/evaluate_render.py \
    --gaussian-output-dynamic-dir "${GAUSSIAN_OUTPUT_DYNAMIC_DIR}" \
    --gaussian-data-dir "${GAUSSIAN_DATA_DIR}" \
    --render-eval-data-dir "${RENDER_EVAL_DATA_DIR}" \
    --human-mask-path "${HUMAN_MASK_PATH}" \
    --results-dir "${RESULTS_DIR}"
