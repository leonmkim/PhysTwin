#!/usr/bin/env bash
# Output-root overrides (Stage C). Defaults preserve upstream paths.
GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR="${GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR:-./gaussian_output_dynamic_white}"
REFERENCE_GAUSSIAN_OUTPUT_DIR="${REFERENCE_GAUSSIAN_OUTPUT_DIR:-./gaussian_output}"
REFERENCE_EXPERIMENTS_DIR="${REFERENCE_EXPERIMENTS_DIR:-experiments}"
# Scene source for gs_render_dynamics.py (-s). Set GAUSSIAN_DATA_DIR to a scratch
# root (e.g. temp_gaussian_data_uv) to avoid rewriting points3D.ply under author data/.
gaussian_data_dir="${GAUSSIAN_DATA_DIR:-./data/gaussian_data}"

# views=("0" "1" "2")
views=("0")

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

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

for scene_name in "${scenes[@]}"; do

    python gs_render_dynamics.py \
        -s "${gaussian_data_dir}/${scene_name}" \
        -m "${REFERENCE_GAUSSIAN_OUTPUT_DIR}/${scene_name}/${exp_name}" \
        --name "${scene_name}" \
        --white_background \
        --output_dir "${GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR}" \
        --reference-experiments-dir "${REFERENCE_EXPERIMENTS_DIR}"

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder "${GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR}/${scene_name}/${view_name}" \
            --video_path "${GAUSSIAN_OUTPUT_DYNAMIC_WHITE_DIR}/${scene_name}/${view_name}.mp4"
    done

done
