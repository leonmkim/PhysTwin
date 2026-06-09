#!/usr/bin/env bash
# PhysTwin segmentation environment helper (source only).
#
#   cd third_party/phystwin
#   source scripts/setup_segmentation_env.sh
#
# Shared cache default (workstation):
#   /mnt/data2/magna/belt_perception/third_party/phystwin_external/
# Scratch fallback:
#   PHYSTWIN_EXTERNAL_ROOT="$(pwd)/temp_external_uv" source scripts/setup_segmentation_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: source this script; do not execute it directly." >&2
  echo "  source scripts/setup_segmentation_env.sh" >&2
  exit 1
fi

_phystwin_seg_env_setup() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"

  if [[ ! -f "${repo_root}/pyproject.toml" ]] \
    || [[ ! -f "${repo_root}/data_process/segment_util_video.py" ]] \
    || [[ ! -f "${repo_root}/process_data.py" ]]; then
    echo "error: source scripts/setup_segmentation_env.sh from the PhysTwin repo root." >&2
    echo "  expected markers: pyproject.toml, process_data.py, data_process/segment_util_video.py" >&2
    echo "  resolved repo_root=${repo_root}" >&2
    return 1
  fi

  export PHYSTWIN_EXTERNAL_ROOT="${PHYSTWIN_EXTERNAL_ROOT:-/mnt/data2/magna/belt_perception/third_party/phystwin_external}"
  export GSAM_ROOT="${GSAM_ROOT:-${PHYSTWIN_EXTERNAL_ROOT}/Grounded-SAM-2}"
  export PHYSTWIN_SEG_CHECKPOINT_DIR="${PHYSTWIN_SEG_CHECKPOINT_DIR:-${PHYSTWIN_EXTERNAL_ROOT}/checkpoints}"

  local sam2_config_dir="${GSAM_ROOT}/sam2/configs/sam2.1"
  local config_link="${repo_root}/configs/sam2.1"
  local checkpoint_dir="${repo_root}/data_process/groundedSAM_checkpoints"
  local sam2_ckpt_src="${PHYSTWIN_SEG_CHECKPOINT_DIR}/sam2.1_hiera_large.pt"
  local gdino_ckpt_src="${PHYSTWIN_SEG_CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
  local sam2_ckpt_link="${checkpoint_dir}/sam2.1_hiera_large.pt"
  local gdino_ckpt_link="${checkpoint_dir}/groundingdino_swint_ogc.pth"
  local config_status="missing"
  local sam2_ckpt_status="missing"
  local gdino_ckpt_status="missing"
  local had_error=0

  if [[ ! -d "${GSAM_ROOT}" ]]; then
  cat >&2 <<EOF
error: Grounded-SAM-2 clone not found at:
  GSAM_ROOT=${GSAM_ROOT}

One-time shared setup (requires explicit approval for network/git clone):

  export PHYSTWIN_EXTERNAL_ROOT=${PHYSTWIN_EXTERNAL_ROOT}
  mkdir -p "\${PHYSTWIN_EXTERNAL_ROOT}"
  git clone https://github.com/IDEA-Research/Grounded-SAM-2.git "\${GSAM_ROOT}"

Per-worktree editable installs (after uv sync preprocessing-core):

  uv pip install -e "\${GSAM_ROOT}"
  uv pip install --no-build-isolation -e "\${GSAM_ROOT}/grounding_dino"
  uv pip install 'transformers<5' yapf timm addict pycocotools

Scratch fallback (single worktree only):

  export PHYSTWIN_EXTERNAL_ROOT="\$(pwd)/temp_external_uv"
  source scripts/setup_segmentation_env.sh

EOF
    return 1
  fi

  if [[ ! -d "${sam2_config_dir}" ]]; then
    echo "error: SAM2 config directory not found at ${sam2_config_dir}" >&2
    return 1
  fi

  mkdir -p "${repo_root}/configs"
  ln -sfn "${sam2_config_dir}" "${config_link}"
  if [[ -L "${config_link}" ]] && [[ -e "${config_link}/sam2.1_hiera_l.yaml" ]]; then
    config_status="ok -> $(readlink "${config_link}")"
  else
    echo "error: failed to create SAM2 config symlink at ${config_link}" >&2
    had_error=1
  fi

  export PYTHONPATH="${GSAM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  mkdir -p "${checkpoint_dir}"

  _phystwin_link_checkpoint() {
    local src="$1"
    local dst="$2"
    local label="$3"
    if [[ -f "${src}" ]]; then
      ln -sfn "${src}" "${dst}"
      if [[ -e "${dst}" ]]; then
        echo "ok -> ${src}"
        return 0
      fi
      echo "error: failed to link ${label}" >&2
      return 1
    fi
    echo "missing (source not found: ${src})"
    cat >&2 <<EOF
warning: ${label} checkpoint missing.
  expected shared file: ${src}
  worktree symlink target: ${dst}
  download requires explicit approval; when approved, place files in:
    ${PHYSTWIN_SEG_CHECKPOINT_DIR}/
  or run (from PhysTwin repo root, writing into shared checkpoint dir):
    mkdir -p "${PHYSTWIN_SEG_CHECKPOINT_DIR}"
    bash env_install/download_pretrained_models.sh
    # then copy/move *.pt and *.pth into ${PHYSTWIN_SEG_CHECKPOINT_DIR}/
EOF
    return 2
  }

  local ckpt_rc=0
  _phystwin_link_checkpoint "${sam2_ckpt_src}" "${sam2_ckpt_link}" "SAM2" || ckpt_rc=$?
  if [[ ${ckpt_rc} -eq 0 ]]; then
    sam2_ckpt_status="ok"
  elif [[ ${ckpt_rc} -eq 2 ]]; then
    sam2_ckpt_status="missing"
  else
    sam2_ckpt_status="error"
    had_error=1
  fi

  ckpt_rc=0
  _phystwin_link_checkpoint "${gdino_ckpt_src}" "${gdino_ckpt_link}" "GroundingDINO" || ckpt_rc=$?
  if [[ ${ckpt_rc} -eq 0 ]]; then
    gdino_ckpt_status="ok"
  elif [[ ${ckpt_rc} -eq 2 ]]; then
    gdino_ckpt_status="missing"
  else
    gdino_ckpt_status="error"
    had_error=1
  fi

  cat <<EOF
PhysTwin segmentation environment summary
  repo_root:                 ${repo_root}
  PHYSTWIN_EXTERNAL_ROOT:    ${PHYSTWIN_EXTERNAL_ROOT}
  GSAM_ROOT:                 ${GSAM_ROOT}
  PHYSTWIN_SEG_CHECKPOINT_DIR: ${PHYSTWIN_SEG_CHECKPOINT_DIR}
  PYTHONPATH prefix:         ${GSAM_ROOT}
  configs/sam2.1:            ${config_status}
  checkpoint sam2.1_hiera_large.pt: ${sam2_ckpt_status}
  checkpoint groundingdino_swint_ogc.pth: ${gdino_ckpt_status}
EOF

  if [[ ${had_error} -ne 0 ]]; then
    return 1
  fi
  return 0
}

_phystwin_seg_env_setup
_phystwin_seg_env_rc=$?
return ${_phystwin_seg_env_rc}
