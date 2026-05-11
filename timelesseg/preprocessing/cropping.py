from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_fill_holes
import torch
from torch.nn import functional as F


def get_bbox_from_mask(mask: np.ndarray) -> list[list[int]]:
    """
    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def bounding_box_to_slice(bounding_box: list[list[int]]):
    return tuple([slice(*i) for i in bounding_box])

def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    """
    :param data: NDIM + 1 (4d if image is 3d!)
    :return: the mask is True where the data is nonzero in any of the c channels
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)

def crop_to_nonzero(data: np.ndarray, seg: np.ndarray = None, nonzero_label = -1) -> \
    tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    :param data: NDIM + 1 (4d if image is 3d!)
    :param seg: NDIM + 1 (4d if image is 3d!)
    :param nonzero_label: this will be written into the segmentation map. It controls the areas that will be used during normalization.
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]

    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox

def int_bbox(bbox):
    bbox2 = [[int(i) for i in row] for row in bbox]  # Convert all values to int
    if bbox2 != bbox:  # Directly compare the original and casted versions
        raise RuntimeError(f"Invalid bbox encountered. Cannot be safely cast to int: {bbox}. Casting result: {bbox2}")
    return bbox2

def crop_and_pad_nd(
    image: torch.Tensor | np.ndarray,
    bbox: list[list[int]],
    pad_value=0,
    pad_mode: str = 'constant',
    allow_hacky_np_workaround_for_reflect: bool = True
) -> torch.Tensor | np.ndarray:
    """
    Crops a bounding box directly specified by bbox, adhering to the half-open interval [start, end).
    If the bounding box extends beyond the image boundaries, the cropped area is padded
    to maintain the desired size. Initial dimensions not included in bbox remain unaffected.

    CAREFUL: When allow_hacky_np_workaround_for_reflect=True, pad_mode is 'reflect', the input is torch and the
    padding exceeds the dimension of the tensor in the respective dimension (for example paddding with 80 on a dim with
     shape 32) torch will not be able to perform this operation. We fall back to CPU (!!) numpy padding for this!

    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73

    Parameters:
    - image: N-dimensional torch.Tensor, np.ndarray, or blosc2.ndarray.NDArray representing the image.
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions.
            Each dimension follows the half-open interval [start, end).
    - pad_value: Value used for padding when bbox extends beyond image boundaries.
    - pad_mode: Padding mode, one of 'constant', 'reflect', or 'replicate' (alias for 'edge').
    - allow_hacky_np_workaround_for_reflect: If True, will perform numpy workaround if torch.pad doesn't work. Only
    happening if pad_mode is 'reflect'

    Returns:
    - Cropped and padded patch of the requested bounding box size, as the same type as `image`.
    """

    assert pad_mode in ['constant', 'reflect', 'replicate', 'edge'], "Unsupported pad_mode."

    # make sure bounding boxes are int and not uint. Otherwise we may get underflow
    bbox = int_bbox(bbox)

    # Determine the number of dimensions to crop based on bbox
    crop_dims = len(bbox)
    img_shape = image.shape
    num_dims = len(img_shape)

    # Initialize the crop and pad specifications for each dimension
    slices = []
    padding = []
    output_shape = list(img_shape[:num_dims - crop_dims])  # Initial dimensions remain as in the original image
    target_shape = output_shape + [max_val - min_val for min_val, max_val in bbox]  # Half-open interval

    # Iterate through dimensions, applying bbox to the last `crop_dims` dimensions
    for i in range(num_dims):
        if i < num_dims - crop_dims:
            # For initial dimensions not covered by bbox, include the entire dimension
            slices.append(slice(None))
            padding.append([0, 0])
            output_shape.append(img_shape[i])
        else:
            # For dimensions specified in bbox, directly use the min and max bounds
            dim_idx = i - (num_dims - crop_dims)  # Index within bbox

            min_val = bbox[dim_idx][0]
            max_val = bbox[dim_idx][1]  # This is exclusive by definition

            # Check if the bounding box is completely outside the image bounds
            if max_val <= 0 or min_val >= img_shape[i]:
                # If outside bounds, return an empty array or tensor of the target shape
                if isinstance(image, torch.Tensor):
                    return torch.zeros(target_shape, dtype=image.dtype, device=image.device)
                elif isinstance(image, np.ndarray):
                    return np.zeros(target_shape, dtype=image.dtype)
                else:
                    raise TypeError('Unsupported image type %s' % str(type(image)))

            # Calculate valid cropping ranges within image bounds (half-open interval)
            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])  # Exclusive upper bound
            slices.append(slice(valid_min, valid_max))

            # Calculate padding needed for this dimension
            pad_before = max(0, -min_val)
            pad_after = max(0, max_val - img_shape[i])
            padding.append([pad_before, pad_after])

            # Define the shape based on the bbox range in this dimension
            output_shape.append(max_val - min_val)

    # Crop the valid part of the bounding box
    cropped = image[tuple(slices)]

    # Apply padding to the cropped patch
    if np.any(padding):
        # torch will not allow reflection padding if the amount of padding exceeds the shape in that dimension. Numpy will happily do that. I am annoyed. Implement a numpy fallback for that
        was_torch = False
        if allow_hacky_np_workaround_for_reflect and pad_mode == 'reflect' and isinstance(image, torch.Tensor):
            for d in range(-len(padding), 0):
                if max(padding[d]) < cropped.shape[d]:
                    device = cropped.device
                    cropped = cropped.cpu().numpy()
                    was_torch = True
                    print('used numpy workaround')
                    break

        if isinstance(cropped, torch.Tensor):
            if pad_mode == 'edge':
                pad_mode = 'replicate'
            if pad_mode in ['replicate', 'reflect']:
                # pytorch is weird: https://github.com/pytorch/pytorch/issues/147506
                # getting NotImplementedError: Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now

                # when padding a 2d array because batch/color channel are expected. This is not documented.
                # wonky workaround that may work, or not. Adding fake dimension. This is fiddly because our
                # image/cropped may have more dimensions than we want to pad and may even have too many dimensions for
                # torch to be happy

                # ok so pytorch works most reliably if the length of the padding is 2 less than the ndim of cropped
                while padding[0] == [0, 0] and len(padding) > cropped.ndim - 2:
                    padding = padding[1:]
                # now check whether cropped.ndim is too small
                n_fake_dims = len(padding) + 2 - cropped.ndim
                if n_fake_dims > 0:
                    for i in range(n_fake_dims):
                        cropped.unsqueeze_(0)
                # check if we need to add fake padding
                while len(padding) < cropped.ndim - 2:
                    padding = [0, 0] + padding
                assert cropped.ndim < 6, 'Torch padding with replicate/reflect works with 3D images at most'

            flattened_padding = [p for sublist in reversed(padding) for p in sublist]  # Flatten in reverse order for PyTorch
            try:
                padded_cropped = F.pad(cropped, flattened_padding, mode=pad_mode, value=pad_value)
            except Exception as e:
                print('Failed torch pad')
                print('cropped', cropped.shape)
                print('cropped device', cropped.device)
                print('cropped type', cropped.dtype)
                print('flattened_padding', flattened_padding)
                print('pad mode', pad_mode)
                print('pad value', pad_value)
                print('image shape', img_shape)
                print('bbox', bbox)
                raise e

            if pad_mode in ['replicate', 'reflect'] and n_fake_dims > 0:
                for i in range(n_fake_dims):
                    padded_cropped.squeeze_(0)
        elif isinstance(cropped, np.ndarray):
            if pad_mode == 'replicate':
                pad_mode = 'edge'
            if pad_mode == 'edge' or pad_mode == 'reflect':
                kwargs = {}
            else:
                kwargs = {'constant_values': pad_value}
            pad_width = [(p[0], p[1]) for p in padding]
            padded_cropped = np.pad(cropped, pad_width=pad_width, mode=pad_mode, **kwargs)
            if was_torch:
                padded_cropped = torch.from_numpy(padded_cropped).to(device)
        else:
            raise ValueError(f'Unsupported image type {type(image)}')
    else:
        padded_cropped = cropped

    return padded_cropped

if __name__ == "__main__":
    import sys
    from timelesseg.data_generation.image import load_nifti
    import timeit
    mask = load_nifti(sys.argv[1])
    _mask = mask.data

    # theirs is 10 times faster than mine lol
    # print(timeit.timeit(lambda: combined_acvl(_mask), number=1000)) 4.773196374997497
    # print(timeit.timeit(lambda: get_nonzero_slicer(_mask), number=1000)) 45.687010374997044