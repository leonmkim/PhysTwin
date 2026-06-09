#!/usr/bin/env bash
# Output-root overrides (Stage C). Defaults preserve upstream paths.
#
# GAUSSIAN_DATA_DIR is the scene source root passed to gs_train/gs_render (-s).
# INTERP_POSES_OUTPUT_DIR is optional; when set, generate_interp_poses.py writes
# interp_poses.pkl under that scratch root instead of mutating author data/.
#
# If INTERP_POSES_OUTPUT_DIR is scratch (e.g. temp_gaussian_data_uv), you must
# either set GAUSSIAN_DATA_DIR to the same scratch root or seed that scratch tree
# from reference ./data/gaussian_data first. Downstream training reads -s from
# GAUSSIAN_DATA_DIR only; it will not pick up poses written elsewhere.
GAUSSIAN_OUTPUT_DIR="${GAUSSIAN_OUTPUT_DIR:-./gaussian_output}"
GAUSSIAN_OUTPUT_VIDEO_DIR="${GAUSSIAN_OUTPUT_VIDEO_DIR:-./gaussian_output_video}"
GAUSSIAN_DATA_DIR="${GAUSSIAN_DATA_DIR:-./data/gaussian_data}"
INTERP_POSES_OUTPUT_DIR="${INTERP_POSES_OUTPUT_DIR:-}"

# scenes=("double_lift_cloth_1" "double_lift_cloth_3" "double_lift_sloth" "double_lift_zebra"
#         "double_stretch_sloth" "double_stretch_zebra"
#         "rope_double_hand"
#         "single_clift_cloth_1" "single_clift_cloth_3"
#         "single_lift_cloth" "single_lift_cloth_1" "single_lift_cloth_3" "single_lift_cloth_4"
#         "single_lift_dinosor" "single_lift_rope" "single_lift_sloth" "single_lift_zebra"
#         "single_push_rope" "single_push_rope_1" "single_push_rope_4"
#         "single_push_sloth"
#         "weird_package")

scenes=("double_stretch_sloth")

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

interp_args=(--gaussian-data-dir "${GAUSSIAN_DATA_DIR}")
if [[ -n "${INTERP_POSES_OUTPUT_DIR}" ]]; then
    interp_args+=(--interp-poses-output-dir "${INTERP_POSES_OUTPUT_DIR}")
fi
python ./gaussian_splatting/generate_interp_poses.py "${interp_args[@]}"

# Iterate over each folder
for scene_name in "${scenes[@]}"; do
    echo "Processing: $scene_name"

    # Training
    python gs_train.py \
        -s "${GAUSSIAN_DATA_DIR}/${scene_name}" \
        -m "${GAUSSIAN_OUTPUT_DIR}/${scene_name}/${exp_name}" \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid'

    # Rendering
    python gs_render.py \
        -s "${GAUSSIAN_DATA_DIR}/${scene_name}" \
        -m "${GAUSSIAN_OUTPUT_DIR}/${scene_name}/${exp_name}"

    # Convert images to video
    python gaussian_splatting/img2video.py \
        --image_folder "${GAUSSIAN_OUTPUT_DIR}/${scene_name}/${exp_name}/test/ours_10000/renders" \
        --video_path "${GAUSSIAN_OUTPUT_VIDEO_DIR}/${scene_name}/${exp_name}.mp4"
done
