import numpy as np
import SimpleITK as sitk

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.longitudinal.ciao import CiaoBaseline
from batchgeneratorsv2.transforms.spatial.reg_errors import RegistrationErrorsTransform


from timelesseg.data_stuff import LABELS
from timelesseg.dataloading.dataloader import DataLoader
from timelesseg.data_augmentation.rotation import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from timelesseg.config import get_configs

from ..multithreaded_augmenter import NonDetMultiThreadedAugmenter
from .test_training import TestTrainer


def get_training_transforms(
    patch_size: tuple[int] | list[int] | np.ndarray,
    rotation_for_DA: RandomScalar,
    deep_supervision_scales: list | tuple | None,
    mirror_axes: tuple[int],
    do_dummy_2d_data_aug: bool,
    p_cross_sectional: float = 0.,
    use_mask_for_norm: list[bool] = None
) -> BasicTransform:

    # set jth entry p_per_channel to 0 if you want to turn off that channel's chance of suffering that augmentation
    # Sampling from a uniform distribution(0, 1) never gives a number less than 0
    transforms = []
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None

    transforms.append(
        SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,  # , mode_seg='nearest'
            is_seg_per_channel=[False, True] # treat channel 1 as a seg
        )
    )

    if do_dummy_2d_data_aug:
        transforms.append(Convert2DTo3DTransform())

    # these are taken from:
    # https://github.com/BBillot/SynthSR/blob/9e63935503d97a433fcf7082b9f69abc5e7a1ddd/SynthSR/labels_to_image_model.py#L233
    # No original ideas, just stealing
    translation_fraction = 1 / np.array(patch_size_spatial).max()
    rotation_bounds = .5 * np.pi / 180
    transforms.append(RandomTransform(
        RegistrationErrorsTransform(
            reference_channel=0,
            rotation_bounds=rotation_bounds,
            translation_fraction=translation_fraction,
            scale_fraction=0, # no scaling
            resampling_mode='nearest',
            padding_mode='zeros'
        ), apply_probability=1. # for testing
    ))


    if p_cross_sectional > 0.:
        transforms.append(
            # assume baseline is in last channel
            CiaoBaseline(p_per_channel = [0, p_cross_sectional])
        )

    transforms.append(
        RemoveLabelTansform(-1, 0)
    )

    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    return ComposeTransforms(transforms)

class DummyTrainer(TestTrainer):

    def get_dataloaders(self, patch_size):

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

        # so that predictor knows which axes to do mirroring on
        self.allowed_mirror_axes = mirror_axes

        training_dataset, validation_dataset = self._get_datasets()

        training_transforms = get_training_transforms(patch_size, rotation_for_DA,
                                                      self.deep_supervision_scales,
                                                      mirror_axes, do_dummy_2d_data_aug,
                                                      self.config.p_cross_sectional)

        validation_transforms = None

        # I just work on binary seg
        training_dataloader = DataLoader(training_dataset, 2,
                                         initial_patch_size, patch_size, LABELS,
                                         self.config.oversample_fg_probability,
                                         transforms=training_transforms)
        validation_dataloader = DataLoader(validation_dataset, self.config.batch_size,
                                           patch_size, patch_size, LABELS,
                                           self.config.oversample_fg_probability,
                                           transforms=validation_transforms)

        mt_training_dataloader = NonDetMultiThreadedAugmenter(training_dataloader, None, num_processes=self.config.num_processes,
                                                              num_cached=max(6, self.config.num_processes // 2), seeds=None,
                                                              pin_memory=self.device.type in ['cuda', 'mps'], wait_time=0.002)
        mt_validation_dataloader = NonDetMultiThreadedAugmenter(validation_dataloader, None, self.config.num_processes//2,
                                                                num_cached=max(3, self.config.num_processes // 4), seeds=None,
                                                                pin_memory=self.device.type in ["cuda", 'mps'], wait_time=0.002)

        _ = next(mt_training_dataloader)
        _ = next(mt_validation_dataloader)

        return mt_training_dataloader, mt_validation_dataloader

    def run_training(self):
        self.on_training_start()
        num_cross_sectional = 0
        ntotal = 0
        for epoch in range(self.current_epoch, self.num_epochs):
            self._on_training_epoch_start()
            for batch_id in range(self.config.training_iters_per_epoch):
                bs = self.config.batch_size
                data, target = self.train_step(next(self.dataloader_train))
                assert data.shape[0] == bs
                for b in range(bs):
                    im, baseline = data[b, 0], data[b, 1]
                    seg = target[b][0]
                    assert im.shape == baseline.shape == seg.shape

                    sitk.WriteImage(sitk.GetImageFromArray(im), self.output_folder + '/test_im_batch_%i.nii.gz' % (batch_id + b))
                    sitk.WriteImage(sitk.GetImageFromArray(baseline), self.output_folder + '/test_baseline_batch_%i.nii.gz' % (batch_id + b))
                    sitk.WriteImage(sitk.GetImageFromArray(seg), self.output_folder + '/test_seg_batch_%i.nii.gz' % (batch_id + b))

                    baseline = baseline.astype(bool, copy=False)
                    empty = np.zeros_like(baseline, dtype=bool)
                    if np.array_equal(baseline, empty):
                        num_cross_sectional += 1
                    ntotal += 1
        print(num_cross_sectional / ntotal)


if __name__ == "__main__":
    arch_kwargs, training_config, preprocessing_config = get_configs('resunet', 36,
                                                                     num_epochs=2,
                                                                     training_iters_per_epoch=5)
    tr = DummyTrainer(training_config,
                     arch_kwargs,
                     'cpu',
                     'test-training')
    tr.run_training()