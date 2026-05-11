import SimpleITK as sitk
import numpy as np

from ..dataloader import DataLoaderDownsamplesChannel1
from ..dataset import Dataset
from timelesseg.data_augmentation.transforms import get_training_transforms
from timelesseg.data_augmentation.rotation import configure_rotation_dummyDA_mirroring_and_inital_patch_size

def _get_transforms(patch_size, deep_supervision_scales):
    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes
    ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

    transforms = get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug, use_mask_for_norm=None)
    return transforms, initial_patch_size

def main(data_folder: str):
    dataset = Dataset(data_folder)
    patch_size = (192, 224, 128)
    deep_supervision_scales = [[1.] * 3, [0.5] * 3, [0.25] * 3, [0.125] * 3]
    transforms, initial_patch_size = _get_transforms(patch_size, deep_supervision_scales)
    dataloader = DataLoaderDownsamplesChannel1(
        dataset, 5, initial_patch_size, patch_size, all_labels=[0, 1], oversample_foreground_percent=0.5, transforms=transforms,
        deep_supervision_scales=deep_supervision_scales
    )
    for _ in range(10):
        batch = next(dataloader)
        data = batch['data']
        target = batch['target']
        baseline = batch['baseline']
        print('We have gotten the following keys!', batch['keys'], end='.\n')
        for j, k in enumerate(batch['keys']):

            sitk.WriteImage(sitk.GetImageFromArray(data[j][0].numpy()), 'data_channel_0_key_%s.nii.gz' % k)
            sitk.WriteImage(sitk.GetImageFromArray(data[j][1].numpy().astype(np.uint8)), 'data_channel_1_key_%s.nii.gz' % k)

            assert len(target) == len(baseline)
            for l, (t, b) in enumerate(zip(target, baseline)):
                sitk.WriteImage(
                    sitk.GetImageFromArray(t[j][0].numpy().astype(np.uint8)),
                    f'target_dim_{l}_key_{k}.nii.gz'
                )
                sitk.WriteImage(
                    sitk.GetImageFromArray(b[j][0].numpy().astype(np.uint8)),
                    f'baseline_dim_{l}_key_{k}.nii.gz'
                )


if __name__ == "__main__":
    import sys
    main(sys.argv[1])