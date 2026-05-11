#!/bin/bash

set -e

ims=(
    "P22_T1_FLAIR.nii.gz"
    "P22_T1_T2.nii.gz"
    "P22_T1_T1.nii.gz"
)
weights=(
    "0.5"
    "0.3"
    "0.2"
)

# can be one of [cpu, cuda, mps]
DEVICE="mps"

command=(
    "python3" "entrypoint.py"
    "-i" "${ims[@]/#/examples/data/P22/T1/}"
    "--device" "$DEVICE"
    "-o" "example-cross-sectional"
    "-w" "${weights[@]}"
    "-npe" "3" # three processes for segmentation export
    "-npp" "3" # three processes for preprocessing all data
)

# I use conda, but this is up to you
PYTORCH_ENABLE_MPS_FALLBACK="1" conda run -n timelesseg-test --live-stream "${command[@]}"
