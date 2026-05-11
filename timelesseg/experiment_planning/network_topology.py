from copy import deepcopy
import numpy as np

from timelesseg.typing import ArrayLike

def get_shape_must_be_divisible_by(net_numpool_per_axis: ArrayLike) -> ArrayLike:
    return 2 ** np.array(net_numpool_per_axis)

def pad_shape(shape: ArrayLike, must_be_divisible_by: int | ArrayLike) -> ArrayLike:
    """
    pads shape so that it is divisible by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp

def get_pool_and_conv_props(
    spacing: ArrayLike,
    patch_size: ArrayLike,
    min_feature_map_size: int,
    max_numpool: int
):
    """
    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck -  defaults to 4!
    :param max_numpool: this is usually set to 999999 (essentially infinite)
    :return:
    """
    # todo review this code
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    # first set of strides is set to 1 per dimension since this will be applied to the input image.
    # thus, we want to extract features also at the input dimensions/spacing
    pool_op_kernel_sizes = [[1] * len(spacing)]
    conv_kernel_sizes = []

    # pool == downsampling
    num_pool_per_axis = [0] * dim

    # these are initialized to 1 but may vary well be set to 3 in the first iteration
    kernel_size = [1] * dim

    while True:
        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]

        # find axis that are within factor of 2 within smallest spacing
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]

        # max_numpool constraint
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1:
            if current_size[valid_axes_for_pool[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes_for_pool) < 1:
            break

        # now we need to find kernel sizes
        # kernel sizes are initialized to 1. They are successively set to 3 when their associated axis becomes within
        # factor 2 of min_spacing. Once they are 3 they remain 3
        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            else:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    def _to_tuple(lst):
        return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, _to_tuple(pool_op_kernel_sizes), _to_tuple(conv_kernel_sizes), tuple(patch_size), must_be_divisible_by