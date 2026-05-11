from copy import deepcopy
import numpy as np

from timelesseg.preprocessing.resampling import compute_new_shape
from timelesseg.models import get_net_from_name
from timelesseg.configs import get_default_config_from_strides

from .network_topology import get_pool_and_conv_props
from .utils import (
    initialize_patch_size,
    estimate_network_memory_consumption
)


MIN_FEATURE_MAP_DIM = 4
REFERENCE_BS = 2
MIN_BATCH_SIZE = 2


def get_reference_values(net_name: str) -> tuple[int, int]:

    match net_name.lower():
        # these are taken from Default Planner
        case 'unet' | 'plainconvunet':
            (
                reference_consum,
                reference_gb
            ) = 560_000_000, 8

        # these are taken from ResEncUNetPlannerM
        case 'resencunet':
            (
                reference_consum,
                reference_gb
            ) = 680_000_000, 8

        # these are taken from ResEncUNetPlannerXL
        case 'resunet' | 'mednext':
            (
                reference_consum,
                reference_gb
            ) = 3_600_000_000, 40

        case _:
            raise NotImplementedError(net_name)

    return reference_consum, reference_gb


def get_network_topology(
    target_spacing,
    median_image_shape,
    patch_size = None,
    min_feature_map_dim: int = MIN_FEATURE_MAP_DIM
):

    if patch_size is None:
        patch_size = initialize_patch_size(target_spacing, median_image_shape)

    (
        network_num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        shape_must_be_divisible_by
    ) = get_pool_and_conv_props(target_spacing, patch_size, min_feature_map_dim, max_numpool=999999)

    return network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, shape_must_be_divisible_by

def is_valid_experiment_re_consumption(
    estimate_consum,
    available_mem,
    reference_consum: int,
    reference_mem: int,
    reference_batch_size: int = REFERENCE_BS
) -> tuple[bool, int]:
    tmp = reference_consum * (available_mem / reference_mem)
    valid = tmp >= 2 * (estimate_consum / reference_batch_size)
    batch_size = round((reference_consum / estimate_consum) * reference_batch_size)
    return valid, batch_size


def plan_experiment(preprocessing_config,
                    net: str,
                    target_memory: int = 64):

    target_spacing = preprocessing_config.target_spacing
    assert preprocessing_config._spacing_has_been_transposed

    new_shapes = [compute_new_shape(j, i, target_spacing) for i, j \
                  in zip(preprocessing_config.dataset_fingerprint['spacings'], preprocessing_config.dataset_fingerprint['shapes_after_cropping'])]

    new_median_shape = np.median(new_shapes, 0)
    new_median_shape = new_median_shape[preprocessing_config.transpose_forward]

    n_voxels_in_dataset = np.prod(new_median_shape) * len(preprocessing_config.dataset_fingerprint['spacings'])

    (
        network_num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size,
        shape_must_be_divisible_by
    ) = get_network_topology(target_spacing, new_median_shape)

    net_kwargs = get_default_config_from_strides(net=net, kernel_sizes=conv_kernel_sizes, strides=pool_op_kernel_sizes, input_channels=2)
    network_class = get_net_from_name(net)
    estimated_memory_consumption = estimate_network_memory_consumption(network_class, patch_size, net_kwargs)

    reference_consum, reference_gb = get_reference_values(net)
    is_valid_config, batch_size = is_valid_experiment_re_consumption(estimated_memory_consumption, target_memory,
                                                                     reference_consum, reference_gb)

    while not is_valid_config:
        # get axis where patch size is greater compared to median image shape
        axis_to_be_reduced = np.argsort([p / s for p, s in zip(patch_size, new_median_shape)])[-1]

        patch_size = list(patch_size)
        tmp = deepcopy(patch_size)
        tmp[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

        _, _, _, _, shape_must_be_divisible_by =  get_network_topology(target_spacing, new_median_shape, tmp)

        patch_size[axis_to_be_reduced] -= shape_must_be_divisible_by[axis_to_be_reduced]

        # now recompute topology
        (
            network_num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            patch_size,
            shape_must_be_divisible_by
        ) = get_network_topology(target_spacing, new_median_shape, patch_size)

        net_kwargs = get_default_config_from_strides(net=net, kernel_sizes=conv_kernel_sizes, strides=pool_op_kernel_sizes, input_channels=2)
        estimated_memory_consumption = estimate_network_memory_consumption(network_class, patch_size, net_kwargs)

        is_valid_config, batch_size = is_valid_experiment_re_consumption(estimated_memory_consumption, target_memory,
                                                                         reference_consum, reference_gb)

    # cap batch size to be at least 2 but not be greater than 5% of the total number of voxels per batch
    batch_size = max(
        min(batch_size, round(n_voxels_in_dataset * 0.05 / np.prod(patch_size))),
        MIN_BATCH_SIZE
    )

    return patch_size, net_kwargs, batch_size, network_class


def plan_entrypoint():
    import argparse
    from timelesseg.configs.preprocessing import get_preprocessing_config_from_dataset_fingerprint

    parser = argparse.ArgumentParser()
    parser.add_argument('-dfp', '--dataset_fingerprint', type=str,
                        default='training_data/raw/dataset_fingerprint.json')
    parser.add_argument('-net', choices=['unet', 'plainconvunet', 'resunet'],
                        default='unet')
    parser.add_argument('-tmem', '--target_memory', type=int,
                        default=8)

    args = parser.parse_args()

    prep_config = get_preprocessing_config_from_dataset_fingerprint(args.dataset_fingerprint)
    return plan_experiment(prep_config, args.net, args.target_memory)

if __name__ == "__main__":
    print(plan_entrypoint())