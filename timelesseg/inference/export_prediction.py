import torch
import numpy as np

from timelesseg.typing import ArrayLike
from timelesseg.utils import write_pickle

from timelesseg.preprocessing.resampling import resample_data_or_seg_to_shape
from timelesseg.preprocessing.cropping import int_bbox
from timelesseg.configs.preprocessing import PrepConfig

from timelesseg.io import rw

from .logits_to_probabilities import convert_logits_to_segmentation


def resample_probabilities(data: np.ndarray | torch.Tensor,
                           new_shape: ArrayLike,
                           current_spacing: ArrayLike,
                           new_spacing: ArrayLike) -> np.ndarray | torch.Tensor:
    kwargs = {
            "is_seg": False,
            "order": 1,
            "order_z": 0,
            "force_separate_z": None,
    }
    return resample_data_or_seg_to_shape(data, new_shape, current_spacing, new_spacing, **kwargs)


def insert_crop_into_image(
    image: torch.Tensor | np.ndarray,
    crop: torch.Tensor | np.ndarray,
    bbox: list[list[int]]
) -> torch.Tensor | np.ndarray:
    """
    Inserts a cropped patch back into the original image at the position specified by bbox.
    If the bounding box extends beyond the image boundaries, only the valid portions are inserted.
    If the bounding box lies entirely outside the image, the original image is returned.

    Parameters:
    - image: Original N-dimensional torch.Tensor or np.ndarray to which the crop will be inserted.
    - crop: Cropped patch of the image to be reinserted. May have additional dimensions compared to bbox.
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions of the crop in the original image.

    Returns:
    - image: The original image with the crop reinserted at the specified location (modified in-place).
    """
    # make sure bounding boxes are int and not uint. Otherwise we may get underflow
    bbox = int_bbox(bbox)

    # Ensure that bbox only applies to the last len(bbox) dimensions of crop and image
    num_dims = len(image.shape)
    crop_dims = len(crop.shape)
    bbox_dims = len(bbox)

    if crop_dims < bbox_dims:
        raise ValueError("Bounding box dimensions cannot exceed crop dimensions.")

    # Validate that non-cropped leading dimensions match between image and crop
    leading_dims = num_dims - bbox_dims
    if image.shape[:leading_dims] != crop.shape[:leading_dims]:
        raise ValueError("Leading dimensions of crop and image must match.")

    # Check if the bounding box lies completely outside the image bounds for each cropped dimension
    for i in range(bbox_dims):
        min_val, max_val = bbox[i]
        dim_idx = leading_dims + i  # Corresponding dimension in the image

        if max_val <= 0 or min_val >= image.shape[dim_idx]:
            # If completely out of bounds in any dimension, return the original image
            return image

    # Prepare slices for inserting the crop into the original image
    image_slices = []
    crop_slices = []

    # Iterate over all dimensions, applying bbox only to the last len(bbox) dimensions
    for i in range(num_dims):
        if i < leading_dims:
            # For leading dimensions, use entire dimension (slice(None)) and validate shape
            image_slices.append(slice(None))
            crop_slices.append(slice(None))
        else:
            # For dimensions specified by bbox, calculate the intersection with image bounds
            dim_idx = i - leading_dims
            min_val, max_val = bbox[dim_idx]

            crop_start = max(0, -min_val)  # Start of the crop within the valid area
            image_start = max(0, min_val)  # Start of the image where the crop will be inserted
            image_end = min(max_val, image.shape[i])  # Exclude upper bound by using max_val directly

            # Adjusted range for insertion
            crop_end = crop_start + (image_end - image_start)

            # Append slices for both image and crop insertion ranges
            image_slices.append(slice(image_start, image_end))
            crop_slices.append(slice(crop_start, crop_end))

    # Insert the valid part of the crop back into the original image
    if isinstance(image, torch.Tensor):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    elif isinstance(image, np.ndarray):
        image[tuple(image_slices)] = crop[tuple(crop_slices)]
    else:
        raise ValueError(f"Unsupported image type {type(image)}")

    return image


def revert_cropping_on_probabilities(
    probabilities_array: np.ndarray | torch.Tensor,
    bbox: list[list[int]],
    original_shape: ArrayLike
) -> np.ndarray | torch.Tensor:
    factory = np.zeros if isinstance(probabilities_array, np.ndarray) else torch.zeros
    probabilities_reverted_cropping = factory((probabilities_array.shape[0], *original_shape), dtype=probabilities_array.dtype)
    # if not self.has_regions: WHY???
    probabilities_reverted_cropping[0] = 1
    probabilities_reverted_cropping = insert_crop_into_image(probabilities_reverted_cropping, probabilities_array, bbox)
    return probabilities_reverted_cropping


def convert_predicted_logits_to_segmentation_with_correct_shape(
    predicted_logits: torch.Tensor | np.ndarray,
    properties: dict,
    preprocessing_config: PrepConfig,
    num_threads_torch: int,
    return_probabilities: bool = False
):
    """
    Here we have to revert the operations performed during preprocessing, in inverse order.

    The operations are:
    3 - Resampling
    2 - Cropping
    1 - Transposing
    (normalization omitted for obvious reasons)
    """

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    ## Revert resampling (resample from target_spacing to original spacing)
    # Since properties['spacing'] is "untransposed", first transpose it
    spacing_transposed = [properties['spacing'][i] for i in preprocessing_config.transpose_forward]
    current_spacing = preprocessing_config.target_spacing
    predicted_logits = resample_probabilities(
        predicted_logits,
        properties['shape_after_cropping'],
        current_spacing,
        spacing_transposed
    )

    # the following comment is deprecated... resampling_fn_probabilities can take in a np.ndarray or 
    # a torch.Tensor, but it always returns a np.ndarray...
    # [DEPRECATED] return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    segmentation, predicted_probabilities = convert_logits_to_segmentation(predicted_logits, return_probabilities)
    del predicted_logits

    # I'm tired of all the ndarray and Tensor conversions, we should convert everything to ndarray
    # and work from there... This doesn't make any sense. It's already what's happening,
    # but the comments are misleading and make me feel scared to touch anything.
    # TODO: make everything except insert_crop_into_image into ndarray only
    assert (
        isinstance(segmentation, np.ndarray) and (predicted_probabilities is None or isinstance(predicted_probabilities, np.ndarray))
    ), str(type(segmentation)) + '\n' + str(type(predicted_probabilities))

    ## Revert cropping (put segmentation in bbox)
    segmentation_reverted_cropping = np.zeros(properties['shape_before_cropping'], dtype=np.uint8)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    ## Revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(preprocessing_config.transpose_backward)

    if not return_probabilities:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


    predicted_probabilities = revert_cropping_on_probabilities(predicted_probabilities,
                                                               properties['bbox_used_for_cropping'],
                                                               properties['shape_before_cropping'])

    # see lines 153-164
    # predicted_probabilities = predicted_probabilities.cpu().numpy()

    # Revert transpose on probabilities
    predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in preprocessing_config.transpose_backward])
    torch.set_num_threads(old_threads)
    return segmentation_reverted_cropping, predicted_probabilities


def export_prediction_from_logits(
    predicted_array: np.ndarray | torch.Tensor,
    properties: dict,
    preprocessing_config: PrepConfig,
    save_probabilities: bool,
    outfile: str | None,
    file_ending: str = '.nii.gz',
    num_threads_torch: int = 8
):
    """
    :param outfile: has to be truncated! Not including the extension (i.e., `.nii.gz`)
    """

    ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_array,
                                                                      properties,
                                                                      preprocessing_config,
                                                                      num_threads_torch,
                                                                      save_probabilities)
    del predicted_array

    # return segmentation and probabilities (or None, depending on whether we want them)
    if outfile is None:
        return ret

    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(outfile + '.npz', probabilities=probabilities_final)
        write_pickle(properties, outfile + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw.write_seg(segmentation_final, outfile + file_ending, properties)