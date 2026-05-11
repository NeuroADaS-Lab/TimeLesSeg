from dataclasses import dataclass
from torch import nn

from timelesseg.models.utils import Net, features_per_stage
from timelesseg.data_stuff import LABELS

from .abstract import Config


INITIAL_NUM_FEATURES = 32
MAX_NUM_FEATURES = 320

@dataclass
class ArchKwargs(Config):
    """Maybe this class could be an attribute of TrainingConfig?"""
    input_channels: int
    kernel_sizes: list[int | list[int] | tuple[int]]
    strides: list[int | list[int] | tuple[int]]
    n_stages: int
    features_per_stage: list[int]
    conv_op: nn.Module | None
    conv_bias: bool
    n_conv_per_stage: list[int]
    n_conv_per_stage_decoder: list[int]
    num_classes: int
    norm_op: nn.Module | None
    norm_op_kwargs: dict | None
    nonlin: nn.Module | None
    nonlin_kwargs: dict | None
    dropout_op: nn.Module | None
    dropout_op_kwargs: dict | None
    deep_supervision: bool
    expansion_ratio_per_stage: list[int | list[int] | tuple[int]] | None = None

    def __post_init__(self):
        assert (
            len(self.kernel_sizes) == len(self.strides) == self.n_stages
        )
        # this is a fix for compatibility when using Unet or non-expansion-ratio-per-stage nets
        if self.expansion_ratio_per_stage is None:
            delattr(self, 'expansion_ratio_per_stage')
        else:
            assert (
                len(self.expansion_ratio_per_stage) == (len(self.n_conv_per_stage) + len(self.n_conv_per_stage_decoder))
            )

def get_default_config_from_strides(
    net: str,
    kernel_sizes,
    strides,
    **overrides
) -> dict:
    num_stages = len(strides)
    net_kwargs = get_default_config(net, num_stages)
    net_kwargs |= {'kernel_sizes': kernel_sizes, 'strides': strides, **overrides}
    return net_kwargs

def get_default_config(net: str | Net, n_stages: int) -> dict:

    match net.lower():
        case Net.MEDNEXT:
            kwargs = _get_default_config_mednext(n_stages)

        case Net.UNet:
            kwargs = _get_default_config_unet(n_stages)

        case Net.ResEncUNet:
            raise NotImplementedError
            return _get_default_config_resencunet(n_stages)

        case Net.ResUNet:
            kwargs = _get_default_config_resunet(n_stages)

        case _:
            raise ValueError(net)

    return kwargs


def _get_default_config_mednext(n_stages: int) -> dict:
    default_arch_kwargs = {
        'input_channels': 2,
        'kernel_sizes': [[3] * 3] * n_stages,
        'strides': [[1] * 3] + [[2] * 3] * (n_stages - 1),
        'n_stages': n_stages,
        'features_per_stage': features_per_stage(INITIAL_NUM_FEATURES, n_stages, MAX_NUM_FEATURES),
        'expansion_ratio_per_stage': [2] * (2 * n_stages - 1),
        'conv_op': nn.Conv3d,
        'conv_bias': True,
        'n_conv_per_stage': [2] * n_stages,
        'n_conv_per_stage_decoder': [2] * (n_stages - 1),
        'num_classes': len(LABELS),
        'norm_op': nn.GroupNorm,
        'norm_op_kwargs': None,
        'nonlin': nn.GELU,
        'nonlin_kwargs': None,
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'deep_supervision': True
    }
    return default_arch_kwargs


def _get_default_config_unet(n_stages: int) -> dict:
    default_arch_kwargs = {
        'input_channels': 2,
        'kernel_sizes': [[3] * 3] * n_stages,
        'strides': [[1] * 3] + [[2] * 3] * (n_stages - 1),
        'n_stages': n_stages,
        'features_per_stage': features_per_stage(INITIAL_NUM_FEATURES, n_stages, MAX_NUM_FEATURES),
        'conv_op': nn.Conv3d,
        'conv_bias': True,
        'n_conv_per_stage': [2] * n_stages,
        'n_conv_per_stage_decoder': [2] * (n_stages - 1),
        'num_classes': len(LABELS),
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'dropout_op': nn.Dropout3d,
        'dropout_op_kwargs': {'p': 0.5},
        'deep_supervision': True
    }
    return default_arch_kwargs

def _get_default_config_resunet(n_stages: int) -> dict:
    """We do not set block, which means we'll use BasicBlockD"""
    default_arch_kwargs = {
        'input_channels': 2,
        'kernel_sizes': [[3] * 3] * n_stages,
        'strides': [[1] * 3] + [[2] * 3] * (n_stages - 1),
        'n_stages': n_stages,
        'features_per_stage': features_per_stage(INITIAL_NUM_FEATURES, n_stages, MAX_NUM_FEATURES),
        'conv_op': nn.Conv3d,
        'conv_bias': True,
        'n_conv_per_stage': [2] * n_stages,
        'n_conv_per_stage_decoder': [2] * (n_stages - 1),
        'num_classes': len(LABELS),
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'dropout_op': nn.Dropout3d,
        'dropout_op_kwargs': {'p': 0.5},
        'deep_supervision': True
    }
    return default_arch_kwargs