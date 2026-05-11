from typing import Iterable
import numpy as np

from timelesseg.typing import ArrayLike, Number
from timelesseg.utils import save_json, join, run_parallel
from timelesseg.io import rw
from timelesseg.dataloading.utils import generate_iterable_with_fnames

from .cropping import crop_to_nonzero


# What does nnUNet extract?
# 1 - shape before and after cropping to nonzero -- not actually what the shape before cropping was,
# but actually how much cropping reduces shape.
# 2 - spacing
# 3 - some stats per case (that is, mean median min max 99.5 and 0.05 percentiles of each image)
# 4 - some (10_000) voxels that are appended to a very big list to then compute stats for the whole dataset

def collect_intensities_first_channel(
    data: np.ndarray,
    seg: np.ndarray,
    percentiles: ArrayLike,
    seed: int = 999,
    num_voxels: int = 10_000
):
    """We don't care about the second channel, since that is just a seg"""
    assert not np.isnan(data).any() and data.ndim == seg.ndim == 4
    rs = np.random.RandomState(seed)

    fg_voxels_im = data[0][seg[0] > 0]
    num_fg_voxels = len(fg_voxels_im)

    intensity_stats = None
    if num_fg_voxels > 0:
        p = np.percentile(fg_voxels_im, percentiles)
        intensity_stats = {
            'mean': np.mean(fg_voxels_im),
            'max': fg_voxels_im.max(),
            'min': fg_voxels_im.min(),
            'percentiles': {str(percentiles[pp]): p[pp] for pp in range(len(percentiles))}
        }

    voxels_to_return = rs.choice(fg_voxels_im, size=num_voxels)
    return voxels_to_return, intensity_stats

def analyze_case(images_paths: Iterable[str],
                 seg_path: str,
                 percentiles: ArrayLike,
                 num_samples: int = 10_000):

    data, properties = rw.read_images(images_paths)
    seg, _ = rw.read_seg(seg_path)

    spacing: ArrayLike = properties['spacing']
    shape_before_cropping = data.shape[1:]

    data_cropped, seg_cropped, bbox = crop_to_nonzero(data, seg)
    shape_after_cropping = data_cropped.shape[1:]
    relative_size_after_cropping = np.prod(shape_after_cropping) / np.prod(shape_before_cropping)

    some_voxels, intensity_stats = collect_intensities_first_channel(
        data_cropped,
        seg_cropped,
        percentiles = percentiles,
        num_voxels=num_samples
    )

    return some_voxels, intensity_stats, shape_after_cropping, relative_size_after_cropping, spacing

def run_and_save(
    input_folder: str,
    output_folder: str,
    num_processes: int,
    total_voxels_to_sample: int = 10e7,
    percentiles: ArrayLike = (50, 0.05, 99.5),
    timeout: Number = 2
):

    dataset = generate_iterable_with_fnames(input_folder, allow_no_seg=False, file_ending='.nii.gz')
    ncases = len(dataset)
    nvoxels_per_case = int(total_voxels_to_sample / ncases)

    all_args = [
        (case_dict['images'], case_dict['seg'], percentiles, nvoxels_per_case)
        for case_dict in dataset.values()
    ]

    # r = []
    # with multiprocessing.get_context("spawn").Pool(num_processes) as p:
    #     for k in dataset:
    #         these_args = [(dataset[k]['images'], dataset[k]['seg'], percentiles, nvoxels_per_case),]
    #         r.append(p.starmap_async(analyze_case, these_args))

    #     remaining = list(range(len(dataset)))
    #     # p is pretty nifti. If we kill workers they just respawn but don't do any work.
    #     # So we need to store the original pool of workers.
    #     workers = [j for j in p._pool]

    #     while len(remaining) > 0:
    #         all_alive = all([j.is_alive() for j in workers])
    #         if not all_alive:
    #             raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
    #                                 'OK jokes aside.\n'
    #                                 'One of your background processes is missing. This could be because of '
    #                                 'an error (look for an error message) or because it was killed '
    #                                 'by your OS due to running out of RAM. If you don\'t see '
    #                                 'an error message, out of RAM is likely the problem. In that case '
    #                                 'reducing the number of workers might help')
    #         done = [i for i in remaining if r[i].ready()]
    #         remaining = [i for i in remaining if i not in done]
    #         sleep(0.1)

    results = run_parallel(analyze_case, all_args, num_processes, timeout)
    fg_voxels_images, intensity_stats, shapes_after_cropping, relative_sizes_after_cropping, spacings = zip(*results)

    # some bounds considerations: if total_voxels_to_sample is divisible by n_cases, then we'll have
    # exactly fg_voxels_images total_voxels_to_sample/ncases * ncases == total_voxels_to_sample.
    # However, worst case scenario,  total_voxels_to_sample - int(total_voxels_to_sample/ncases) * ncases == ncases - 1
    # e.g. if total_voxels_to_sample=19, ncases is 4: fg_voxels_images=int(19/4)*4=4*4=16
    # then (16 - 19) / 4 <= 1
    fg_voxels_images = np.concatenate(fg_voxels_images)
    # assert len(fg_voxels_images) - total_voxels_to_sample <= 1

    if fg_voxels_images.size < total_voxels_to_sample - ncases:
        print(f"Warning: Expected ~{total_voxels_to_sample} voxels, but only got {fg_voxels_images.size}. "
            "Some segmentations might be empty or too small.")

    _percentiles = np.percentile(fg_voxels_images, percentiles)
    intensity_stats_global = {
        'mean': float(np.mean(fg_voxels_images)),
        'std': float(np.std(fg_voxels_images)),
        'min': float(fg_voxels_images.min()),
        'max': float(fg_voxels_images.max()),
        **{'percentile_' + str(percentiles[pp]): _percentiles[pp] for pp in range(len(percentiles))}
    }

    fingerprint = {
        'spacings': spacings,
        'median_relative_size_after_cropping': np.median(relative_sizes_after_cropping, axis=0),
        'shapes_after_cropping': shapes_after_cropping,
        'intensity_stats_full_dataset': intensity_stats_global,
        'rw_used': rw.__class__.__name__
    }

    save_json(fingerprint, join(output_folder, 'dataset_fingerprint.json'))

if __name__ == "__main__":

    run_and_save(
        'training_data/raw',
        'training_data/raw',
        16
    )