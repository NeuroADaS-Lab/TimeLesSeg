import numpy as np
from torch import nn

from timelesseg.typing import ArrayLike
from timelesseg.models.utils import features_per_stage


ANISOTROPY_THRESHOLD = 3

def determine_target_spacing(spacings: list[ArrayLike], shapes: list[ArrayLike]) -> ArrayLike:
    assert len(spacings) == len(shapes)
    spacings = np.vstack(spacings)
    shapes = np.vstack(shapes)

    target_spacing = np.percentile(spacings, 50, axis=0)
    target_shape = np.percentile(shapes, 50, axis=0)

    worst_spacing_axis = np.argmax(target_spacing)
    other_axes = [i for i in range(len(target_spacing)) if i != worst_spacing_axis]
    other_spacings = [target_spacing[i] for i in other_axes]
    other_shapes = [target_shape[i] for i in other_axes]

    has_aniso_spacing = target_spacing[worst_spacing_axis] > (ANISOTROPY_THRESHOLD * max(other_spacings))
    has_aniso_voxels = target_shape[worst_spacing_axis] * ANISOTROPY_THRESHOLD < min(other_shapes)

    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = spacings[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        # don't let the spacing of that axis get higher than the other axes
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        target_spacing[worst_spacing_axis] = target_spacing_of_that_axis

    return target_spacing


def determine_transpose(target_spacing: ArrayLike, suppress_transpose: bool = False):
    if suppress_transpose:
        return [0, 1, 2], [0, 1, 2]

    max_spacing_axis = np.argmax(target_spacing)
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    transpose_backward = [np.argwhere(np.array(transpose_forward) == i)[0][0] for i in range(3)]
    return transpose_forward, transpose_backward


def initialize_patch_size(target_spacing: ArrayLike, dataset_median_shape: ArrayLike) -> np.ndarray:
    aspect_ratio = 1 / np.array(target_spacing)
    desired_patch_size = np.array([256] * 3)
    initial_patch_size = np.round(aspect_ratio * ((desired_patch_size ** 3) / np.prod(aspect_ratio)) ** (1 / 3))
    return np.minimum(initial_patch_size, dataset_median_shape)

def estimate_network_memory_consumption(
    network_class: nn.Module,
    patch_size: ArrayLike,
    network_kwargs: dict
) -> int:
    initialized_net = network_class(**network_kwargs)
    return initialized_net.compute_conv_feature_map_size(patch_size)

if __name__ == "__main__":
    from timelesseg.models.MedNeXT import MedNeXT

    inital_ps = initialize_patch_size([0.5, 0.5, 0.5], [256, 256, 256])
    print(estimate_network_memory_consumption(
        MedNeXT,
        [256] * 3,
        {'input_channels': 2,
        'num_stages': 6,
        'features_per_stage': features_per_stage(32, 6, 320),
        'expansion_ratio_per_stage': 4,
        'conv_op': nn.Conv3d,
        'kernel_sizes': 3,
        'strides': [1] + [2] * 5,
        'conv_bias': True,
        'n_conv_per_stage': 2,
        'num_classes': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_op':  nn.GroupNorm,
        'norm_op_kwargs':  None,
        'nonlin':  nn.GELU,
        'nonlin_kwargs':  None,
        'deep_supervision':  True}
    ))

    # 22457720832