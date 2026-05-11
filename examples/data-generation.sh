#!/bin/bash

PMI="3" # number of times each lesion mask is initially augmented
IMI="3" # number of images generated per segmentation
FLMI="2" # number of prior timepoints generated per lesion mask

command=(
    "python3" "-m" "timelesseg.data_generation.Dataset_Gen"
    "--lesion_masks" "examples/data/patient1_study2_t1w_to_mni_space_gif/patient1_lesion_mask.nii.gz"
    "--parcellations" "examples/data/patient1_study2_t1w_to_mni_space_gif/patient1_Parcellation.nii.gz"
    "-o" "example-data-generation"
    "--pseudo_mask_iters" "$PMI"
    "--synthseg_iters" "$IMI"
    "--fake_lesion_mask_iters" "$FLMI"
)

conda run -n test-synthseg --live-stream "${command[@]}"