from timelesseg.utils import maybe_mkdir, join

from timelesseg.dataloading.dataloader import DataLoaderDownsamplesChannel1
from timelesseg.dataloading.dataset import Dataset
from timelesseg.data_augmentation.transforms import get_training_transforms, get_validation_transforms
from timelesseg.data_augmentation.rotation import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from timelesseg.trainer.plotting import plot_batched_segmentations



def _get_transforms(patch_size, deep_supervision_scales):
    (
        rotation_for_DA,
        do_dummy_2d_data_aug,
        initial_patch_size,
        mirror_axes
    ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

    transforms = get_training_transforms(
        patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug, p_cross_sectional=0.3, use_mask_for_norm=None
    )
    return transforms, initial_patch_size

def _get_val_transforms(patch_size, deep_supervision_scales):

    transforms = get_validation_transforms(deep_supervision_scales)

    return transforms

def main(data_folder: str):
    dataset = Dataset(data_folder)
    patch_size = (192, 224, 128)
    deep_supervision_scales = [[1.] * 3, [0.5] * 3, [0.25] * 3, [0.125] * 3]
    transforms, initial_patch_size = _get_transforms(patch_size, deep_supervision_scales)
    dataloader = DataLoaderDownsamplesChannel1(
        dataset, 5, initial_patch_size, patch_size, all_labels=[0, 1], oversample_foreground_percent=0.5, transforms=transforms,
        deep_supervision_scales=deep_supervision_scales
    )

    out_folder = 'test_plotting'
    for i in range(10):
        this_folder = join(out_folder, f'iter_{i}')
        maybe_mkdir(this_folder)
        batch = next(dataloader)
        data = batch['data']
        target = batch['target'][0]
        plot_batched_segmentations(data, target, batch['keys'], this_folder)


def main_validation(validation_data_folder: str):
    dataset = Dataset(validation_data_folder)
    patch_size = (192, 224, 128)
    deep_supervision_scales = [[1.] * 3, [0.5] * 3, [0.25] * 3, [0.125] * 3]
    val_transforms = _get_val_transforms(patch_size, deep_supervision_scales)
    dataloader = DataLoaderDownsamplesChannel1(
        dataset, 2, patch_size, patch_size, [0, 1], 0.5, transforms=val_transforms, deep_supervision_scales=deep_supervision_scales
    )

    out_folder = 'test_plotting_val'
    for i in range(10):
        this_folder = join(out_folder, f'iter_{i}')
        maybe_mkdir(this_folder)
        batch = next(dataloader)
        data = batch['data']
        target = batch['target'][0]
        plot_batched_segmentations(data, target, batch['keys'], this_folder)


if __name__ == "__main__":
    # main('training_data/preprocessed')
    main_validation('validation_data/preprocessed')