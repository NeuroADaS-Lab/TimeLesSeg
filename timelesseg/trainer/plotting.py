import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from timelesseg.utils import join, maybe_mkdir
from timelesseg.data_generation.image import save_nifti


COLOR_CYCLE = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)


def select_slice_to_plot2(image: np.ndarray, segmentation: np.ndarray) -> int:
    """
    From nnUNet. I KNOW!

    image and segmentation are expected to be 3D (or 1, x, y)

    selects the slice with the largest amount of fg (how much percent of each class are in each slice? pick slice
    with highest avg percent)

    we give image so that we can easily replace this function if needed
    """
    classes = [i for i in np.sort(pd.unique(segmentation.ravel())) if i > 0]
    # this creates a N, C array where N is the number of slices in the first dimension
    fg_per_slice = np.zeros((image.shape[0], len(classes)))

    # for each class:
    for i, c in enumerate(classes):
        # get binary 3D mask of that class
        fg_mask = segmentation == c

        # select how many elements are there along 2nd and 3d dimensions. I.e., how
        # many fg voxels are there along each slice. Put that in the ith element of
        # the C dimension of the previously created storage array.
        fg_per_slice[:, i] = fg_mask.sum((1, 2))

        # I don't understand this... You divide by the total sum?
        # I change it. Before:
        # fg_per_slice[:, i] /= fg_per_slice.sum()
        # Now: -- guaranteed to give 1 along each class!
        fg_per_slice[:, i] /= fg_per_slice[:, i].sum(0)

    fg_per_slice = fg_per_slice.mean(1)
    return int(np.argmax(fg_per_slice))


def select_slice_to_plot(
seg: np.ndarray
) -> int:
    # select slice with maximum number of nonzero voxels!
    nonzero_perslice = seg.sum((1, 2))

    f = np.argmax
    if nonzero_perslice.sum() == 0:
        f = np.random.choice

    return int(f(nonzero_perslice))

def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

def _generate_overlay(image: np.ndarray,
                      seg: np.ndarray,
                      mapping: dict,
                      color_cycle: tuple[str, ...] = COLOR_CYCLE,
                      overlay_intensity: float = 0.6):
    """
    image can be 2d greyscale or 2d RGB (color channel in last dimension!)

    Segmentation must be label map of same shape as image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    returned image is scaled to [0, 255] (uint8)!!!
    """
    # create a copy of image
    image = np.copy(image)

    if image.ndim == 2:
        image = np.tile(image[:, :, None], (1, 1, 3))
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        else:
            raise RuntimeError(f'if 3d image is given the last dimension must be the color channels (3 channels). '
                               f'Only 2D images are supported. Your image shape: {image.shape}')
    else:
        raise RuntimeError("unexpected image shape. only 2D images and 2D images with color channels (color in "
                           "last dimension) are supported")

    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    # create output
    if mapping is None:
        uniques = np.sort(pd.unique(seg.ravel()))  # np.unique(segmentation)
        mapping = {i: c for c, i in enumerate(uniques)}

    for l in mapping.keys():
        image[seg == l] += overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))

    # rescale result to [0, 255]
    image = image / image.max() * 255
    return image.astype(np.uint8)


def generate_overlay(
    image: np.ndarray,
    seg: np.ndarray,
    output_file: str,
    baseline: np.ndarray = None,
    overlay_intensity: float = 0.4,
    mapping: dict = None
):
    if baseline is None:
        baseline = np.zeros_like(seg)

    seg *= 2
    seg = np.where(np.logical_and(baseline, seg), baseline, seg)
    if mapping is None:
        mapping = {1: 1, 2: 2}
    overlay = _generate_overlay(image, seg, mapping, overlay_intensity=overlay_intensity)
    plt.imsave(output_file, overlay)


def plot_batched_segmentations(
    data: torch.Tensor,
    prediction: torch.Tensor,
    keys: list[str],
    output_folder: str
):
    """data and prediction should be batched, 5D tensors"""

    if data.shape[0] != prediction.shape[0]:
        raise ValueError('Something went terribly wrong. Different batch sizes between model i/o. '
                         'data.shape: %s; prediction.shape: %s.' % (data.shape, prediction.shape))

    has_baselines = data.shape[1] == 2

    this_affine = np.eye(4)
    maybe_mkdir(output_folder)
    for b, k in enumerate(keys):
        scan = data[b][0].detach().cpu().numpy()
        pred = prediction[b][0].detach().cpu().numpy()

        selected_slice = select_slice_to_plot(pred)

        baseline = None
        if has_baselines:
            baseline = data[b][1].detach().cpu().numpy().astype(np.uint8)
            save_nifti(baseline, join(output_folder, 'case_%s_baseline.nii.gz' % k), affine=this_affine)

            baseline = baseline[selected_slice]

        # save to nifti first since then we fuck up things
        save_nifti(scan, join(output_folder, 'case_%s_scan.nii.gz' % k), affine=this_affine)
        save_nifti(pred.astype(np.uint8), join(output_folder, 'case_%s_prediction.nii.gz' % k), affine=this_affine)

        output_file = join(output_folder, 'case_%s_slice_%i.png' % (k, selected_slice))
        generate_overlay(scan[selected_slice], pred[selected_slice], output_file, baseline)


if __name__ == "__main__":
    from timelesseg.data_generation.image import load_nifti
    test_image = load_nifti('test-data/NEW_LESIONS_IMAGINEM/FIS_017_01_0001.nii.gz').data
    test_seg = load_nifti('test-data/NEW_LESIONS_IMAGINEM/FIS_017_mask_def.nii.gz').data
    x = select_slice_to_plot2(test_image, test_seg)
    assert x == select_slice_to_plot(test_seg)