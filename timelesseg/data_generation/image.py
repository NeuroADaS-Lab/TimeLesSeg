from typing import Union, Tuple
import numpy as np
import nibabel as nib

from timelesseg.typing import Number

class NiftiImage:

    def __init__(self, im: nib.Nifti1Image):
        self._nib_im = im
        self._affine: np.ndarray = im.affine
        self._header: nib.Nifti1Header = im.header
        self._ndim: int = self._header['dim'][0]
        self._spacing: Tuple[Number] = self._header.get('pixdim')[1:self._ndim+1]

    @classmethod
    def from_path(cls, path: str):
        im = nib.load(path)
        return cls(im)

    @property
    def data(self) -> np.ndarray:
        if not hasattr(self, '_data'):
            self._data = np.asanyarray(self.nib_im.dataobj)
        return self._data

    @property
    def ndim(self):
        return self._ndim

    @property
    def spacing(self) -> Tuple[Number]:
        return self._spacing

    @property
    def affine(self) -> np.ndarray:
        return self._affine

    @property
    def header(self) -> nib.Nifti1Header:
        return self._header

    @property
    def ax_codes(self) -> Tuple[str]:
        if not hasattr(self, '_ax_codes'):
            self._ax_codes = nib.aff2axcodes(self._affine)
        return self._ax_codes

    @property
    def shape(self) -> Tuple[int]:
        return self.header.get_data_shape()

    @property
    def nib_im(self):
        return self._nib_im

    @nib_im.setter
    def nib_im(self, new_im: nib.Nifti1Image):
        self._nib_im = new_im
        self._header = new_im.header
        self._affine = new_im.affine
        if getattr(self, '_data', None) is not None:
            self._data = np.asanyarray(new_im.dataobj)

    @property
    def axial_dim(self):
        try:
            return self.ax_codes.index('I')
        except ValueError:
            return self.ax_codes.index('S')

    @property
    def dtype(self):
        return self._nib_im.get_data_dtype()

    def set_dtype(self, dtype):
        new_header = self.header.copy()
        new_header.set_data_dtype(dtype)
        return self.__class__(nib.Nifti1Image(self.data.astype(dtype), self.affine, new_header))

    def save(self, save_path: str) -> None:
        nib.save(self.nib_im, save_path)

def load_nifti(image_path: str) -> NiftiImage:
    return NiftiImage.from_path(image_path)

def save_nifti(
    nifti_or_ndarray: Union[nib.Nifti1Image, np.ndarray],
    save_path: str,
    *,
    affine: np.ndarray = None,
    header = None
) -> None:

    if isinstance(nifti_or_ndarray, np.ndarray):
        assert affine is not None
        nifti_or_ndarray = nib.Nifti1Image(nifti_or_ndarray, affine, header)
    # for compatibility
    elif isinstance(nifti_or_ndarray, NiftiImage):
        nifti_or_ndarray.save(save_path)
        return

    nifti = NiftiImage(nifti_or_ndarray)
    nifti.save(save_path)

if __name__ == "__main__":
    # import sys

    seg = 'test-data/NEW_LESIONS_IMAGINEM/FIS_102_mask_def.nii.gz'
    old = NiftiImage.from_path(seg)
    old_dtype = old.dtype
    old_data = old.data
    assert old_data.dtype == old_dtype

    old_data_to_dtype = old_data.astype(np.uint16)

    new = old.set_dtype(np.uint16)
    new_dtype = new.dtype
    assert new_dtype == np.uint16 == new.data.dtype == old_data_to_dtype.dtype