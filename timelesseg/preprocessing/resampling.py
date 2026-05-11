from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates
from skimage.transform import resize

from timelesseg.typing import ArrayLike


ANISO_THRESHOLD = 3

def get_do_separate_z(spacing: ArrayLike, anisotropy_threshold: int):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: ArrayLike):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def compute_new_shape(
    old_shape: ArrayLike,
    old_spacing: ArrayLike,
    new_spacing: ArrayLike
) -> np.ndarray:
    assert len(old_spacing) == len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def determine_do_sep_z_and_axis(
    force_separate_z: bool,
    current_spacing,
    new_spacing,
    separate_z_aniso_thr = ANISO_THRESHOLD
):
    axis = None
    do_separate_z = False
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if do_separate_z:
            axis = get_lowres_axis(current_spacing)

    else:
        if get_do_separate_z(current_spacing, separate_z_aniso_thr):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)

        elif get_do_separate_z(new_spacing, separate_z_aniso_thr):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)

    if axis is not None:
        if len(axis) == 1:
            axis = axis[0]
        else:
            do_separate_z = False
            axis = None

    return do_separate_z, axis

def determine_do_sep_z_and_axis_old(
    force_separate_z: bool,
    current_spacing,
    new_spacing,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
) -> Tuple[bool, Union[int, None]]:

    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
            axis = None
        else:
            axis = axis[0]

    return do_separate_z, axis


def resize_segmentation(segmentation: np.ndarray, new_shape, order: int = 3):
    """
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    """
    tpe = segmentation.dtype
    assert segmentation.ndim == len(new_shape), "new shape must have same dimensionality as segmentation"

    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)

    reshaped = np.zeros(new_shape, dtype=tpe)
    unique_labels = np.sort(pd.unique(segmentation.ravel()))
    for i, c in enumerate(unique_labels):
        mask = segmentation == c
        reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = c
    return reshaped


def resample_data_or_seg_to_spacing(
    data: np.ndarray,
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
) -> np.ndarray:

    assert data.ndim == 4, "data must be c x y z"

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    new_shape = compute_new_shape(np.array(data.shape[1:]), current_spacing, new_spacing)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg_to_shape(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: ArrayLike,
    current_spacing: ArrayLike,
    new_spacing: ArrayLike,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    assert data.ndim == 4, "data must be c x y z"

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_or_seg(
    data: np.ndarray,
    new_shape: ArrayLike,
    is_seg: bool = False,
    axis: Union[None, int] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out = None
):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if not np.any(shape != new_shape):
        return data

    dtype_out = dtype_out or data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    data = data.astype(float, copy=False)
    if do_separate_z:
        assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
        new_shape_2d = new_shape[[i for i in range(3) if i != axis]]

        for c in range(data.shape[0]):
            tmp = deepcopy(new_shape)
            tmp[axis] = shape[axis]
            reshaped_here = np.zeros(tmp)
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                elif axis == 1:
                    reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                else:
                    reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_here.shape

                # align_corners=False
                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                if not is_seg or order_z == 0:
                    reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                else:
                    unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                    for i, cl in enumerate(unique_labels):
                        reshaped_final[c][np.round(
                            map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                            mode='nearest')) > 0.5] = cl
            else:
                reshaped_final[c] = reshaped_here
    else:
        for c in range(data.shape[0]):
            reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
    return reshaped_final


def resample_data_to_shape_multiple_channels(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: ArrayLike,
    current_spacing: ArrayLike,
    new_spacing: ArrayLike,
    is_seg: Iterable[bool],
    orders: Iterable[int],
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
):
    """
    needed for segmentation export. Stupid, I know
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    assert isinstance(is_seg, Iterable)
    assert isinstance(orders, Iterable)

    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_multiple_channels(data, new_shape, is_seg, orders, axis, do_separate_z, order_z=order_z)
    return data_reshaped


def resample_data_multiple_channels(
    data: np.ndarray,
    new_shape: ArrayLike,
    is_seg: Iterable[bool],
    orders: Iterable[int],
    axis: Union[None, int] = None,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out = None
) -> np.ndarray:
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)

    if not np.any(shape != new_shape):
        return data

    dtype_out = dtype_out or data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    data = data.astype(float, copy=False)
    if do_separate_z:
        assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
        new_shape_2d = new_shape[[i for i in range(3) if i != axis]]
        for c in range(data.shape[0]):
            if is_seg[c]:
                resize_fn = resize_segmentation
                kwargs = OrderedDict()
            else:
                resize_fn = resize
                kwargs = {'mode': 'edge', 'anti_aliasing': False}
            order = orders[c]

            tmp = deepcopy(new_shape)
            tmp[axis] = shape[axis]
            reshaped_here = np.zeros(tmp)
            for slice_id in range(shape[axis]):
                if axis == 0:
                    reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                elif axis == 1:
                    reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                else:
                    reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_here.shape

                # align_corners=False
                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                if not is_seg or order_z == 0:
                    reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                else:
                    unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                    for i, cl in enumerate(unique_labels):
                        reshaped_final[c][np.round(
                            map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                            mode='nearest')) > 0.5] = cl
            else:
                reshaped_final[c] = reshaped_here
    else:
        for c in range(data.shape[0]):
            if is_seg[c]:
                resize_fn = resize_segmentation
                kwargs = OrderedDict()
            else:
                resize_fn = resize
                kwargs = {'mode': 'edge', 'anti_aliasing': False}
            order = orders[c]
            reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
    return reshaped_final


if __name__ == '__main__':
    channel0 = np.random.random((42, 231, 142))
    channel1 = np.random.randint(0, 2, size=(42, 231, 142)).astype(np.float32)
    input_array = np.zeros(shape=(2, 42, 231, 142), dtype=np.float32)
    input_array[0] = channel0
    input_array[1] = channel1
    output_shape = (52, 256, 256)
    out = resample_data_multiple_channels(input_array, output_shape, is_seg=[False, True], orders=[3, 1])

    print(out.shape, input_array.shape)