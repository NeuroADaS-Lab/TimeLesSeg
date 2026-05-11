from typing import Callable, Optional, Iterable
from functools import partial
import numpy as np

from timelesseg.typing import ArrayLike, ToIterableInt
from timelesseg.utils import join, maybe_mkdir, remove, dirname, run_parallel, get_default_num_processes
from timelesseg.io import rw
from timelesseg.dataloading import Dataset, generate_iterable_with_fnames
from timelesseg.configs.preprocessing import get_preprocessing_config_from_dataset_fingerprint
from timelesseg.data_stuff import LABELS, DATASET_FINGERPRINT_PATH, TRAINING_PATH

from .normalization import zscore_norm, no_normalization
from .cropping import crop_to_nonzero
from .resampling import (
    compute_new_shape,
    resample_data_to_shape_multiple_channels,
    resample_data_or_seg_to_shape
)

def _preprocess_case(
    data: np.ndarray,
    properties: dict,
    normalizers_per_channel: Iterable[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]],
    target_spacing: ArrayLike,
    transpose_forward: ToIterableInt,
    resampling_function: Callable,
    resampling_seg_function: Callable,
    seg: np.ndarray = None,
    verbose: bool = True,
    classes: list[int] = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    This function requires the target spacing that will be used throughout the model, as well as any
    transposition that should be required.

    Order of operations:
        1. Transpose (to bring lowres axis to first dimension) - data + seg (if available) + spacing.
        2. Crop to nonzero (generates an OR mask where any of the data channels are non_zero).
        3. Normalization - for MRI, this should be ZScore.
        4. Resampling to target spacing.
        5. If the data has a segmentation - i.e., it is a training case - foreground locations are extracted,
            for later oversampling of foreground regions during training data loading.
    """

    assert len(data) == len(normalizers_per_channel), 'One would expect the same number of channels as normalization functions per channel...'
    has_seg = seg is not None

    data = data.transpose([0, *[i + 1 for i in transpose_forward]])
    if has_seg:
        if classes is None:
            raise ValueError('If a seg is provided, classes cannot be None.')
        seg = seg.transpose([0, *[i + 1 for i in transpose_forward]])

    original_spacing = [properties['spacing'][i] for i in transpose_forward]

    properties['shape_before_cropping'] = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
    properties['shape_after_cropping'] = data.shape[1:]
    properties['bbox_used_for_cropping'] = bbox

    new_shape = compute_new_shape(properties['shape_after_cropping'], original_spacing, target_spacing)

    for c in range(data.shape[0]):
        if verbose:
            print('Normalizing channel %i with scheme: %s' % (c, repr(normalizers_per_channel[c])))
        # seg is 4D due to crop_to_nonzero, give to normalizer as 3D
        data[c] = normalizers_per_channel[c](data[c], seg=seg[0])

    if verbose:
        print('Resampling from old shape %s to new shape %s (target spacing - %s).' % (
            properties["shape_after_cropping"], new_shape, target_spacing
        ))

    data = resampling_function(data, new_shape, original_spacing, target_spacing)
    seg = resampling_seg_function(seg, new_shape, original_spacing, target_spacing)

    properties['shape_after_resampling'] = data.shape[1:]

    seg = seg.astype(np.int8 if seg.max() <= 255. else np.int16)

    if has_seg:
        properties['class_locations'] = sample_foreground_locations_v2(seg,
                                                                       classes_or_regions=classes,
                                                                       verbose=verbose)

    return data, seg, properties


def sample_foreground_locations(seg: np.ndarray,
                                classes: int | Iterable[int],
                                num_samples: int = 10_000,
                                seed: int = 999,
                                verbose: bool = True,
                                min_percentage_covered: float = .01) -> dict:

    rndst = np.random.RandomState(seed)
    class_locations = {}
    foreground_mask = seg != 0
    foreground_coords = np.argwhere(foreground_mask)
    seg = seg[foreground_mask]
    del foreground_mask
    unique_labels = np.unique(seg.ravel())

    if isinstance(classes, int):
        classes = [classes]

    # We don't need more than 1e7 foreground samples. That's insanity. Cap here
    if len(foreground_coords) > 1e7:
        take_every = np.floor(len(foreground_coords) / 1e7)
        # keep computation time reasonable
        if verbose:
            print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
        foreground_coords = foreground_coords[::take_every]
        seg = seg[::take_every]

    for c in classes:
        if c not in unique_labels:
            class_locations[c] = []
            continue
        this_mask = seg == c
        locations_c = foreground_coords[this_mask]
        if not len(locations_c):
            class_locations[c] = []
            continue

        target_num_samples = min(num_samples, len(locations_c))

        target_num_samples = max(target_num_samples, int(np.ceil(len(locations_c) * min_percentage_covered)))

        locations_selected = locations_c[rndst.choice(len(locations_c), size=target_num_samples, replace=False)]
        class_locations[c] = locations_selected

        if verbose:
            print(c, target_num_samples)

        # this speeds up further iterations. To me, since n_classes == 1, it has no effect
        seg = seg[~this_mask]
        foreground_coords = foreground_coords[~this_mask]

    return class_locations


def sample_foreground_locations_v2(seg: np.ndarray,
                                   classes_or_regions: list[int | tuple[int]],
                                   seed: int = 1234,
                                   verbose: bool = True,
                                   max_voxels_to_sample: int = 10_000,
                                   min_coverage_ratio_per_class: float = 0.01):
    """
    :param seg: Integer-based segmentation we have to collect voxels where classes_or_regions appear
    :param classes_or_regions: integer vals to sample
    :return class_locs: dictionary with a {class: array coordinates} mapping
    """
    randstate = np.random.RandomState(seed)

    class_locs: dict[int | tuple[int], np.ndarray] = {}

    def early_exit(class_locs, classes):
        for c in classes:
            k = tuple(int(x) for x in c) if isinstance(c, Iterable) else int(c)
            class_locs[k] = []
        return class_locs

    # collect only voxels of interest
    classes_of_interest = set()
    for c in classes_or_regions:
        if isinstance(c, Iterable):
            these_classes = tuple(int(x) for x in c)
            classes_of_interest.update(these_classes)
        else:
            classes_of_interest.add(int(c))

    classes_of_interest =  np.fromiter(classes_of_interest, dtype=np.int32, count=len(classes_of_interest))
    valid_mask = np.isin(seg, classes_of_interest)
    coords_of_interest = np.argwhere(valid_mask)
    labels_of_interest = seg[valid_mask]
    del valid_mask

    n_interest_voxels = labels_of_interest.size
    if n_interest_voxels == 0:
        return early_exit(class_locs, classes_or_regions)

    order = np.argsort(labels_of_interest, kind='stable')
    coords_of_interest_order = coords_of_interest[order]
    labels_of_interest_order = labels_of_interest[order]

    # get the ranges of each of the integers in labels_of_interest_order
    # [0, 1, 1, 2] --> [1, 1, 2] != [0, 1, 1]
    # However, in the Python/NumPy community, arr[1:] != arr[:-1] is the standard "idiom" 
    # for finding change points (boundaries), so other researchers reading your code might 
    # recognize the intent of the original version slightly faster.
    # changes = np.flatnonzero(np.ediff1d(labels_of_interest_order)) + 1
    changes = np.flatnonzero(labels_of_interest_order[1:] != labels_of_interest_order[:-1]) + 1

    starts = np.r_[0, changes]
    ends = np.r_[changes, n_interest_voxels]

    labels_present_in_seg = labels_of_interest_order[starts]

    # build mapping from classes of interest to positions in ordered flattened segmentation
    labels_to_ranges_in_ordered_seg = {int(l): (int(s), int(e)) for l, s, e in zip(labels_present_in_seg, starts, ends)}

    labels_present_in_seg = set(labels_present_in_seg)
    for c in classes_or_regions:
        is_region = isinstance(c, Iterable)
        key = tuple(x for x in c) if is_region else int(c)
        current_labs: tuple[int] = key if is_region else (key, )

        if not any(l in labels_present_in_seg for l in current_labs):
            class_locs = early_exit(class_locs, [c])
            continue

        # get counts and ranges for the current class/es
        counts: list[int] = []
        ranges = []
        for l in current_labs:
            range_l = labels_to_ranges_in_ordered_seg.get(l)
            if range_l is None:
                continue
            counts.append(-np.subtract(*range_l))
            ranges.append(range_l)

        total_voxels_c = int(sum(counts))
        if total_voxels_c == 0:
            class_locs = early_exit(class_locs, [c])
            continue

        # if you cumsum lengths/counts, you get relative right/upper bounds. Subtract back lengths, you get the starts!
        upper_bounds_c = np.cumsum(counts)
        lower_bounds_c = upper_bounds_c - counts

        # now we will sample some coordinates for the current class or region.
        target_n_to_sample = min(total_voxels_c, max_voxels_to_sample)
        # don't let it get smaller than the min_coverage_ratio
        target_n_to_sample = max(target_n_to_sample, int(np.ceil(min_coverage_ratio_per_class * total_voxels_c)))

        sampled_relative_voxels = randstate.choice(total_voxels_c, target_n_to_sample, replace=False)

        # get in which class in c each sample has landed
        chosen_class = np.searchsorted(upper_bounds_c, sampled_relative_voxels, side='right')

        nth_voxel_per_class = sampled_relative_voxels - lower_bounds_c[chosen_class]
        absolute_starts = np.fromiter((ranges[i][0] for i in chosen_class), dtype=np.int32, count=target_n_to_sample)
        chosen_positions_absolute = absolute_starts + nth_voxel_per_class

        chosen_coordinates = coords_of_interest_order[chosen_positions_absolute]
        class_locs[key] = chosen_coordinates

    return class_locs


def get_normalizers() -> Iterable[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]]:
    normalizers = (
        partial(zscore_norm, use_mask_for_norm=False),
        partial(no_normalization, use_mask_for_norm=False)
    )
    return normalizers

def get_normalizers_inference(masked_norm: bool) -> tuple[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]]:
    normalizers = (
        partial(zscore_norm, use_mask_for_norm=masked_norm),
        partial(no_normalization, use_mask_for_norm=False)
    )
    return normalizers


def get_resamplers() -> tuple[Callable[[np.ndarray, tuple, tuple, tuple], np.ndarray]]:
    shared_kwargs = {'order_z': 0, 'force_separate_z': None}
    # we have a baseline mask (binary array) as second channel. Thus, we resample as a segmentation!
    resampling_data = partial(resample_data_to_shape_multiple_channels, is_seg = [False, True], orders = [3, 1], **shared_kwargs)
    resampling_seg = partial(resample_data_or_seg_to_shape, is_seg=True, order=1, **shared_kwargs)
    return resampling_data, resampling_seg


def preprocess_case(image_paths: Iterable[str],
                    seg_path: str | None,
                    preprocessing_kwargs: dict):
    """
    Use this in inference.

    :param seg_path: can be `None` (inference)
    """

    # sitk crashes if you try sitk.ReadImage(None)
    # Since I want to keep my generate_iterable_from_folder
    # to return None if baseline is not found, I add a filter
    # pass here
    data, properties = rw.read_images(tuple(filter(None, image_paths)))

    # WARNING. I'm messing this up in purpose!
    # I add this check on data to catch where the "raw case" only has one input
    # i.e., where there is no baseline mask! thus we just add an empty mask -- CS case
    if data.shape[0] == 1:
        print('No baseline found in case %s!' % image_paths[0])
        empty_baseline = np.zeros_like(data)
        data = np.concatenate((data, empty_baseline), dtype=data.dtype, axis=0)

    assert data.shape[0] == 2, 'Incorrect input num of inputs! Found %i channels at files: %s' % (data.shape[0], "\n".join(image_paths))

    seg = seg_path
    if seg_path is not None:
        seg, _ = rw.read_seg(seg_path)

    return _preprocess_case(data, properties, seg=seg, **preprocessing_kwargs)


def preprocess_case_and_save(image_paths: Iterable[str],
                             seg_path: str,
                             output_filename: str,
                             preprocessing_kwargs: dict,
                             clean_after: bool):
    """
    Use this to prepare for training.
    """
    data, seg, properties = preprocess_case(image_paths, seg_path, preprocessing_kwargs)
    Dataset.save_case(data, seg, properties, output_filename)
    if clean_after:
        for file in image_paths + (seg_path, ):
            remove(file)


def preprocess(
    data_iterable: dict[str, dict[str, Iterable[str] | str | None]],
    dataset_fingerprint: str,
    output_folder: str,
    num_processes: int = 12,
    clean_after: bool = False,
    timeout: float = 2.,
    classes: list[int] = None
):

    config = get_preprocessing_config_from_dataset_fingerprint(dataset_fingerprint)
    normalizers_per_channel = get_normalizers()
    resampler_data, resampler_seg = get_resamplers()

    kwargs = {
        'target_spacing': config.target_spacing,
        'transpose_forward': config.transpose_forward,
        'normalizers_per_channel': normalizers_per_channel,
        'resampling_function': resampler_data,
        'resampling_seg_function': resampler_seg,
        'classes': classes
    }

    maybe_mkdir(output_folder)

    args_list = [
        (data_iterable[k]['images'], data_iterable[k]['seg'], join(output_folder, 'case_' + k), kwargs, clean_after)
        for k in data_iterable
    ]

    run_parallel(preprocess_case_and_save, args_list, num_processes, timeout)


def preprocess_entrypoint() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_folder', type=str,
                        help='Where all "identifiable" data is.')
    parser.add_argument('-d', '--dataset_fingerprint', type=str,
                        help='Path to dataset fingerprint. Required to know how preprocessing '
                             'should be approached.')

    parser.add_argument('-o', '--output_folder', type=str,
                        help='Self-explanatory. Fuck you.')
    parser.add_argument('-np', '--num_processes', type=int,
                        help='Self-explanatory. F U.')
    parser.add_argument('--clean_after', action='store_true',
                        help='WARNING. Setting this will delete all data in "raw_data_folder" '
                             'after preprocessing has been run. This is here because I have '
                             'a humongous dataset. You probably don\'t... There\'s classes to this shit here.')
    parser.add_argument('--num_classes', type=int,
                        help="TODO: Write")

    args = parser.parse_args()

    # handle some optional args
    args.dataset_fingerprint = args.dataset_fingerprint or DATASET_FINGERPRINT_PATH
    args.output_folder = args.output_folder or TRAINING_PATH
    args.num_processes = args.num_processes or get_default_num_processes()

    if args.num_classes:
        classes = list(range(1, args.num_classes))
    else:
        classes = LABELS[1:] # remove background

    data_iterable = generate_iterable_with_fnames(args.raw_data_folder, allow_no_seg=False)
    preprocess(data_iterable,
               args.dataset_fingerprint,
               args.output_folder,
               args.num_processes,
               args.clean_after,
               timeout=2,
               classes=classes)


if __name__ == "__main__":
    preprocess_entrypoint()
    # images = ('test-out/If_8948.nii.gz', 'test-out/Mb_8948.nii.gz')
    # seg = 'test-out/Mf_8948.nii.gz'
    # data_iterable = {'8948': {'images': images, 'seg': seg}}
    # data, seg, properties = preprocess(data_iterable, dataset_fingerprint='test-out/dataset_fingerprint.json')
    # for c in range(data.shape[0]):
    #     rw.write_image(data[c], output_fname=f'test_{c}.nii.gz', properties=properties)

    # rw.write_seg(seg[0], 'test_seg.nii.gz', properties)