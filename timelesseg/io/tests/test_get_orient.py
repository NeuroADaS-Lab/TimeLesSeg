import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

from timelesseg.dataloading import generate_iterable_with_fnames
from ..utils import _get_orientation_from_direction, _reorient_nib
from ..reader_writer import SitkReaderWriterReorientOnLoad



def get_orient_nibabel(image_path: str) -> str:
    return "".join(
        nib.aff2axcodes(nib.load(image_path).affine)
    )

def get_orient_sitk(image_path: str) -> str:
    return _get_orientation_from_direction(
        sitk.ReadImage(image_path).GetDirection()
    )

def reorient_nibabel_and_stack(*images: str):
    _to_stack = []
    for im in images:
        reoriented = np.asanyarray(_reorient_nib(nib.load(im))[0].dataobj)
        _to_stack.append(reoriented)
    return np.vstack(_to_stack)

def main(data_folder: str):
    dataset = generate_iterable_with_fnames(folder = data_folder)
    for case in dataset:
        print('Doing case %s' % case)
        images = dataset[case]['images']
        for im in images:
            assert get_orient_nibabel(im) == get_orient_sitk(im)

        assert get_orient_nibabel(dataset[case]['seg']) == get_orient_sitk(dataset[case]['seg'])

def test_reorienting(data_folder):
    rw = SitkReaderWriterReorientOnLoad()
    dataset = generate_iterable_with_fnames(folder = data_folder)
    for case in dataset:
        print(case)
        data, props = rw.read_images(dataset[case]['images'])
        _og_orient = props['sitk_stuff']['original_orient']
        print(_og_orient)
        rw.write_image(data[0], 'test_data.nii.gz', properties=props)

        # reload it and check orient is ok
        should_be_og = _get_orientation_from_direction(sitk.ReadImage('test_data.nii.gz').GetDirection())
        assert _og_orient == should_be_og
        os.system('rm test_data.nii.gz')



if __name__ == "__main__":
    test_reorienting('test-out')