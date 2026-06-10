#!/usr/bin/env bash
# PhysTwin shape-prior / TRELLIS environment helper (source only).
#
#   cd third_party/phystwin
#   source scripts/setup_shape_prior_env.sh
#
# Shared cache default (workstation):
#   /mnt/data2/magna/belt_perception/third_party/phystwin_external/
# Scratch fallback:
#   PHYSTWIN_EXTERNAL_ROOT="$(pwd)/temp_external_uv" source scripts/setup_shape_prior_env.sh
#
# Does not download models, load weights, or run generation.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "error: source this script; do not execute it directly." >&2
  echo "  source scripts/setup_shape_prior_env.sh" >&2
  exit 1
fi

_phystwin_shape_prior_env_setup() {
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"

  if [[ ! -f "${repo_root}/pyproject.toml" ]] \
    || [[ ! -f "${repo_root}/process_data.py" ]] \
    || [[ ! -f "${repo_root}/data_process/shape_prior.py" ]]; then
    echo "error: source scripts/setup_shape_prior_env.sh from the PhysTwin repo root." >&2
    echo "  expected markers: pyproject.toml, process_data.py, data_process/shape_prior.py" >&2
    echo "  resolved repo_root=${repo_root}" >&2
    return 1
  fi

  export PHYSTWIN_EXTERNAL_ROOT="${PHYSTWIN_EXTERNAL_ROOT:-/mnt/data2/magna/belt_perception/third_party/phystwin_external}"
  export TRELLIS_ROOT="${TRELLIS_ROOT:-${PHYSTWIN_EXTERNAL_ROOT}/TRELLIS}"
  export PHYSTWIN_SUPERGLUE_WEIGHT_DIR="${PHYSTWIN_SUPERGLUE_WEIGHT_DIR:-${PHYSTWIN_EXTERNAL_ROOT}/superglue_weights}"
  export HF_HOME="${HF_HOME:-${PHYSTWIN_EXTERNAL_ROOT}/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export TORCH_HOME="${TORCH_HOME:-${PHYSTWIN_EXTERNAL_ROOT}/torch_hub}"

  local trellis_link="${repo_root}/data_process/TRELLIS"
  local weights_dir="${repo_root}/data_process/models/weights"
  local trellis_status="missing"
  local superpoint_status="missing"
  local superglue_indoor_status="missing"
  local superglue_outdoor_status="missing"
  local pytorch3d_status="unknown"
  local had_error=0
  local had_warning=0

  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}"

  if [[ -d "${TRELLIS_ROOT}" ]]; then
    ln -sfn "${TRELLIS_ROOT}" "${trellis_link}"
    if [[ -L "${trellis_link}" ]] && [[ -e "${trellis_link}" ]]; then
      trellis_status="ok -> $(readlink "${trellis_link}")"
    else
      echo "error: failed to create TRELLIS symlink at ${trellis_link}" >&2
      trellis_status="error"
      had_error=1
    fi
  else
    trellis_status="missing"
    had_error=1
    cat >&2 <<EOF
error: TRELLIS clone not found at:
  TRELLIS_ROOT=${TRELLIS_ROOT}

One-time shared setup (requires explicit approval for network/git clone):

  export PHYSTWIN_EXTERNAL_ROOT=${PHYSTWIN_EXTERNAL_ROOT}
  mkdir -p "\${PHYSTWIN_EXTERNAL_ROOT}"
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "\${TRELLIS_ROOT}"

Native build (heavy; not run by this helper):

  cd "\${TRELLIS_ROOT}"
  . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

EOF
  fi

  export PYTHONPATH="${TRELLIS_ROOT}:${repo_root}/data_process${PYTHONPATH:+:${PYTHONPATH}}"

  mkdir -p "${weights_dir}"

  _phystwin_link_weight() {
    local filename="$1"
    local src="${PHYSTWIN_SUPERGLUE_WEIGHT_DIR}/${filename}"
    local dst="${weights_dir}/${filename}"
    if [[ -f "${src}" ]]; then
      ln -sfn "${src}" "${dst}"
      if [[ -e "${dst}" ]]; then
        echo "ok -> ${src}"
        return 0
      fi
      echo "error: failed to link ${filename}" >&2
      return 1
    fi
    echo "missing (source not found: ${src})"
    cat >&2 <<EOF
warning: SuperGlue weight missing: ${filename}
  expected shared file: ${src}
  worktree symlink target: ${dst}
  download requires explicit approval; when approved:

    mkdir -p "${PHYSTWIN_SUPERGLUE_WEIGHT_DIR}"
    cd "${PHYSTWIN_SUPERGLUE_WEIGHT_DIR}"
    wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth
    wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth
    wget https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth

EOF
    return 2
  }

  local w_rc=0
  local w_status
  w_status="$(_phystwin_link_weight superpoint_v1.pth)" || w_rc=$?
  if [[ ${w_rc} -eq 0 ]]; then superpoint_status="ok"; elif [[ ${w_rc} -eq 2 ]]; then superpoint_status="missing"; had_warning=1; else superpoint_status="error"; had_error=1; fi

  w_rc=0
  w_status="$(_phystwin_link_weight superglue_indoor.pth)" || w_rc=$?
  if [[ ${w_rc} -eq 0 ]]; then superglue_indoor_status="ok"; elif [[ ${w_rc} -eq 2 ]]; then superglue_indoor_status="missing"; had_warning=1; else superglue_indoor_status="error"; had_error=1; fi

  w_rc=0
  w_status="$(_phystwin_link_weight superglue_outdoor.pth)" || w_rc=$?
  if [[ ${w_rc} -eq 0 ]]; then superglue_outdoor_status="ok"; elif [[ ${w_rc} -eq 2 ]]; then superglue_outdoor_status="missing"; had_warning=1; else superglue_outdoor_status="error"; had_error=1; fi

  if command -v uv >/dev/null 2>&1 && [[ -f "${repo_root}/pyproject.toml" ]]; then
    if (cd "${repo_root}" && env -u PYTHONPATH uv run --no-sync python -c "from pytorch3d.renderer import look_at_view_transform" >/dev/null 2>&1); then
      pytorch3d_status="importable (align render path)"
    else
      pytorch3d_status="not importable"
      had_warning=1
    fi
  else
    pytorch3d_status="skipped (uv not available)"
    had_warning=1
  fi

  cat <<EOF
PhysTwin shape-prior environment summary
  repo_root:                    ${repo_root}
  PHYSTWIN_EXTERNAL_ROOT:       ${PHYSTWIN_EXTERNAL_ROOT}
  TRELLIS_ROOT:                 ${TRELLIS_ROOT}
  HF_HOME:                      ${HF_HOME}
  HUGGINGFACE_HUB_CACHE:        ${HUGGINGFACE_HUB_CACHE}
  TORCH_HOME:                   ${TORCH_HOME}
  PYTHONPATH prefix:            ${TRELLIS_ROOT}:${repo_root}/data_process
  data_process/TRELLIS:         ${trellis_status}
  superpoint_v1.pth:            ${superpoint_status}
  superglue_indoor.pth:         ${superglue_indoor_status}
  superglue_outdoor.pth:        ${superglue_outdoor_status}
  pytorch3d:                    ${pytorch3d_status}
EOF

  if [[ ${had_warning} -ne 0 ]]; then
    cat <<EOF

notes:
  - PyTorch3D wheel (align render path): install without full gaussian group, e.g.
      cd ${repo_root}
      uv pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl
  - SDXL/TRELLIS HF weights are not downloaded by this helper (HF_HOME routes cache only).
  - Shape-prior generation has not been validated yet.
EOF
  fi

  if [[ ${had_error} -ne 0 ]]; then
    return 1
  fi
  if [[ ${had_warning} -ne 0 ]]; then
    return 2
  fi
  return 0
}

_phystwin_shape_prior_env_setup
_phystwin_shape_prior_rc=$?
return ${_phystwin_shape_prior_rc}
