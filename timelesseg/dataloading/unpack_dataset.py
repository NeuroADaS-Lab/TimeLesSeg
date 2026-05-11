import numpy as np
import multiprocessing

from timelesseg.utils import isfile, remove


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, verify_npy: bool = True, remove_npz: bool = False):
    case_name = npz_file[:-4]
    data_npy = case_name + '.npy'
    seg_npy = case_name + '_seg.npy'

    npz_content = np.load(npz_file)

    np.save(data_npy, npz_content['data'])

    if unpack_segmentation:
        np.save(seg_npy, npz_content['seg'])

    if verify_npy:
        try:
            np.load(data_npy, mmap_mode='r')
            if isfile(seg_npy):
                np.load(seg_npy, mmap_mode='r')
        except ValueError as e:
            raise e

    # I don't have any space on disk...
    if remove_npz:
        remove(npz_file)


def unpack_dataset(npz_files: list[str], num_processes, unpack_segmentation, verify_npy: bool, remove_npz: bool):
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(
            _convert_to_npy,
            [
                (npz_file, unpack_segmentation, verify_npy, remove_npz)
                for npz_file in npz_files
            ],
        )