import numpy as np

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
        ), apply_probability=0.2
    ))

    transforms.append(RandomTransform(
        GaussianNoiseTransform(
            noise_variance=(0, 0.1),
            p_per_channel=[1, 0], # turn off channel 1
            synchronize_channels=True
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            synchronize_channels=False,
            synchronize_axes=False,
            p_per_channel=[0.5, 0], # turn off channel 1
            benchmark=True
        ), apply_probability=0.2
    ))
    transforms.append(RandomTransform(
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=[1, 0] # turn off channel 1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        ContrastTransform(
            contrast_range=BGContrast((0.75, 1.25)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=[1, 0] # turn off channel 1
        ), apply_probability=0.15
    ))
    transforms.append(RandomTransform(
        SimulateLowResolutionTransform(
            scale=(0.5, 1),
            synchronize_channels=False,
            synchronize_axes=True,
            ignore_axes=ignore_axes,
            allowed_channels=None,
            p_per_channel=[0.5, 0] # turn off channel 1
        ), apply_probability=0.25
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=[1, 0], # turn off channel 1
            p_retain_stats=1
        ), apply_probability=0.1
    ))
    transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=[1, 0], # turn off channel 1
            p_retain_stats=1
        ), apply_probability=0.3
    ))
    if mirror_axes is not None and len(mirror_axes) > 0:
        transforms.append(
            MirrorTransform(
                allowed_axes=mirror_axes
            )
        )

    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            channel_idx_in_seg=0,
            set_outside_to=0,
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

def get_validation_transforms(
    deep_supervision_scales: list | tuple | None
) -> BasicTransform:

    transforms = []
    transforms.append(
        RemoveLabelTansform(-1, 0)
    )
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return ComposeTransforms(transforms)
