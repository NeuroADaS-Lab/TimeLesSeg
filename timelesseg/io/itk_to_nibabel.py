import numpy as np
import nibabel as nib
import SimpleITK as sitk
from glob import glob


# source
# https://niftynet.readthedocs.io/en/v0.2.1/_modules/niftynet/io/simple_itk_as_nibabel.html
# class SimpleITKAsNibabel(nib.Nifti1Image):
#     """
#     Minimal interface to use a SimpleITK image as if it were
#     a nibabel object. Currently only supports the subset of the
#     interface used by NiftyNet and is read only
#     """

#     def __init__(self, filename):
#         try:
#             self._SimpleITKImage = sitk.ReadImage(filename)
#         except RuntimeError as err:
#             if 'Unable to determine ImageIO reader' in str(err):
#                 raise nib.filebasedimages.ImageFileError(str(err))
#             else:
#                 raise
#         # self._header = SimpleITKAsNibabelHeader(self._SimpleITKImage)
#         affine = make_affine(self._SimpleITKImage)
#         # super(SimpleITKAsNibabel, self).__init__(
#         #     sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)
#         nib.Nifti1Image.__init__(
#             self,
#             sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)


# class SimpleITKAsNibabelHeader(nib.spatialimages.SpatialHeader):
#     def __init__(self, image_reference):
#         super(SimpleITKAsNibabelHeader, self).__init__(
#             data_dtype=sitk.GetArrayViewFromImage(image_reference).dtype,
#             shape=sitk.GetArrayViewFromImage(image_reference).shape,
#             zooms=image_reference.GetSpacing())



def make_affine(simpleITKImage: sitk.Image):
    # get affine transform in LPS
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine


if __name__ == "__main__":
    some_images = 'test-data/MSLesSeg/MSLesSeg_RAW/P*/T*/*T?.nii.gz'
    some_images = glob(some_images)
    if False:
        for im in some_images:
            sitk_image = sitk.ReadImage(im)
            maybe_affine = make_affine(sitk_image)
            nib_affine = nib.load(im).affine
            assert np.allclose(maybe_affine, nib_affine)

    my_image = some_images[0]
    affine = make_affine(sitk.ReadImage(my_image))
    print('Affine:\n', affine)