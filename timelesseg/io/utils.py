import SimpleITK as sitk
import nibabel as nib
import numpy as np


def _reorient_nib(im: nib.Nifti1Image) -> tuple[nib.Nifti1Image, np.ndarray]:
    original_affine = im.affine
    orient = nib.orientations.io_orientation(original_affine)
    reoriented = im.as_reoriented(orient)
    return reoriented, original_affine

def _reorient_sitk(im: sitk.Image, target_orient) -> sitk.Image:
    _f = sitk.DICOMOrientImageFilter()
    _f.SetDesiredCoordinateOrientation(target_orient)
    im = _f.Execute(im)
    return im

# SITK uses LPS convention
_mapping = {
    1: 'L',
    -1: 'R',
    2: 'P',
    -2: 'A',
    3: 'S',
    -3: 'I'
}

def _get_orientation_from_direction(t: tuple[float]) -> str:
    dim = int(np.sqrt(len(t)))
    orients = [None] * dim
    for i in range(dim):
        this_tuple = t[(i*dim):((i+1)*dim)]
        ix_max = np.argmax(np.abs(this_tuple))
        orients[ix_max] = _mapping[np.sign(this_tuple[ix_max]) * (i + 1)]
    return "".join(orients)