from __future__ import annotations
import numpy as np

from timelesseg.utils import join, isfile, write_pickle, load_pickle

from .utils import get_identifiers
from .unpack_dataset import unpack_dataset


class Dataset:
    suffix: str = '.npz'
    suffix_props: str = '.pkl'
    suffix_uncompressed: str = '.npy'
    def __init__(
        self,
        folder: str
    ):
        super().__init__()
        self.folder = folder
        self.identifiers: list[str] = get_identifiers(folder, (self.suffix, self.suffix_uncompressed))

    def load_case(self, identifier: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Both data and seg are returned as 4D."""
        identifier = 'case_' + identifier

        data_uncompressed_file = join(self.folder, identifier + self.suffix_uncompressed)
        if isfile(data_uncompressed_file):
            data = np.load(data_uncompressed_file, mmap_mode='r')
        else:
            data = np.load(join(self.folder, identifier + self.suffix))['data']

        seg_uncompressed_file = join(self.folder, identifier + '_seg' + self.suffix_uncompressed)
        if isfile(seg_uncompressed_file):
            seg = np.load(seg_uncompressed_file, mmap_mode='r')
        else:
            seg = np.load(join(self.folder, identifier + self.suffix))['seg']

        properties = load_pickle(join(self.folder, identifier + self.suffix_props))
        return data, seg, properties


    @staticmethod
    def save_case(
        data: np.ndarray,
        seg: np.ndarray,
        properties: dict,
        output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + Dataset.suffix, data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + Dataset.suffix_props)


    def unpack_dataset(self, num_processes: int, remove_npz: bool):
        npz_files = list(map(lambda idd: join(self.folder, f'case_{idd}{self.suffix}'), self.identifiers))
        unpack_dataset(npz_files, num_processes, unpack_segmentation=True, verify_npy=True, remove_npz=remove_npz)


if __name__ == "__main__":
    d = Dataset('training_data/preprocessed')
    assert d.identifiers == list(map(lambda x: str(x).rjust(5, '0'), range(1, 16875+1)))

    data, seg, _ = d.load_case('00001')
    assert data.ndim == seg.ndim == 4