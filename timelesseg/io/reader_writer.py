from typing import Iterable, Tuple
from functools import partial
import numpy as np
import SimpleITK as sitk

from timelesseg.singleton import Singleton
from .utils import _get_orientation_from_direction, _reorient_sitk


class SitkReaderWriter(metaclass=Singleton):
    supported_file_endings = [
        '.nii.gz',
        '.nrrd',
        '.mha',
        '.gipl'
    ]

    @staticmethod
    def _check_all_same(input_list):
        if len(input_list) == 1:
            return True
        else:
            # compare all entries to the first
            return np.allclose(input_list[0], input_list[1:])

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if i.shape != input_list[0].shape or not np.allclose(i, input_list[0]):
                return False
        return True

    def read_images(self, image_fnames: Iterable[str]) -> Tuple[np.ndarray, dict]:
        """Images are returned as 4D!"""
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)

            assert npy_image.ndim == 3, 'wtf? %s' % f
            npy_image = npy_image[None]
            spacings_for_nnunet.append(list(spacings[-1])[::-1])

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """Segs are returned as 4D!"""
        return self.read_images((seg_fname, ))

    def write_image(self, data: np.ndarray, output_fname: str, properties: dict, target_dtype = np.float32) -> None:
        assert data.ndim == 3
        itk_image = sitk.GetImageFromArray(data.astype(target_dtype, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        sitk.WriteImage(itk_image, output_fname, True)


class SitkReaderWriterReorientOnLoad(SitkReaderWriter, metaclass=Singleton):
    target_orient: str = 'RAS'
    reorienter = partial(_reorient_sitk, target_orient=target_orient)

    def read_images(self, image_fnames: Iterable[str]) -> Tuple[np.ndarray, dict]:
        """Images are returned as 4D!"""
        images = []
        spacings = []
        origins = []
        directions = []
        orients = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)

            # store original orientation and reorient
            orients.append(_get_orientation_from_direction(itk_image.GetDirection()))
            itk_image = self.reorienter(itk_image)

            # now collect everything post reorient
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)

            assert npy_image.ndim == 3, 'wtf? %s' % f
            npy_image = npy_image[None]
            spacings_for_nnunet.append(list(spacings[-1])[::-1])

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not all(orients[0] == o for o in orients[1:]):
            print('ERROR! Not all input images have the same orientation!')
            print('Orientations:')
            print(orients)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print('WARNING! Not all input images have the same origin!')
            print('Origins:')
            print(origins)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(directions):
            print('WARNING! Not all input images have the same direction!')
            print('Directions:')
            print(directions)
            print('Image files:')
            print(image_fnames)
            print('It is up to you to decide whether that\'s a problem. You should run nnUNetv2_plot_overlay_pngs to verify '
                  'that segmentations and data overlap.')
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a '
                  'bug. Please report!')
            print('spacings_for_nnunet:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0],
                'original_orient': orients[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), dict


    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """Segs are returned as 4D!"""
        return self.read_images((seg_fname, ))


    def write_image(self, data: np.ndarray, output_fname: str, properties: dict, target_dtype = np.float32) -> None:
        assert data.ndim == 3
        itk_image = sitk.GetImageFromArray(data.astype(target_dtype, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        # after setting all the post-reorient stuff, reorient back to original orient
        itk_image = _reorient_sitk(itk_image, properties['sitk_stuff']['original_orient'])

        sitk.WriteImage(itk_image, output_fname, True)


    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, 'segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])

        # after setting all the post-reorient stuff, reorient back to original orient
        itk_image = _reorient_sitk(itk_image, properties['sitk_stuff']['original_orient'])

        sitk.WriteImage(itk_image, output_fname, True)

if __name__ == "__main__":
    test_tuple = (-0.06978551800810884, -0.01851637726696271, -0.9973901570892877,
                  0.9960471045376788, -0.05637405266306358, -0.068644971028431,
                  0.05495586427713073, 0.9982380026623159, -0.022377276284483338)
    print(_get_orientation_from_direction(test_tuple))
    import nibabel as nib
    im = nib.load('SynthDataset/If_0006.nii.gz')
    print(nib.aff2axcodes(im.affine))
