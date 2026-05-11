from torch import nn
from timelesseg.models.PlainUNet.unet import (
    ResidualUNet,
    ResidualEncoderUNet,
    BottleneckD
)

from timelesseg.models.MedNeXT import (
    MedNeXT
)


def main():
    input_shape = (2, 2,) + (256,) * 3

    in_channels = input_shape[1]
    n_stages = 6
    features_per_stage = [32, 64, 128, 256, 320, 320]
    conv_op = nn.Conv3d
    kernel_sizes = [[3] * 3 for _ in range(n_stages)]
    initial_strides = [[1] * 3] + [[2] * 3 for _ in range(n_stages - 1)]
    n_blocks_per_stage=[2] * n_stages
    n_conv_per_stage_decoder=[2] * (n_stages - 1)

    resenc_unet = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=initial_strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=2,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True
    )
    print('Residual encoder Unet Basic Block: ', end="")
    print("{:,}".format(resenc_unet.compute_conv_feature_map_size(input_shape[2:])))
    # 5,753,905,152

    resenc_unet_bottleneck = ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=initial_strides,
        n_blocks_per_stage=n_blocks_per_stage,
        num_classes=2,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True,
        block=BottleneckD,
        bottleneck_channels=[f * 4 for f in features_per_stage]
    )
    print('Residual encoder Unet Bottleneck Block: ', end="")
    print("{:,}".format(resenc_unet_bottleneck.compute_conv_feature_map_size(input_shape[2:])))
    # 20,730,781,696

    res_unet = ResidualUNet(
        in_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        initial_strides,
        n_blocks_per_stage,
        2,
        n_conv_per_stage_decoder,
        True,
        nn.InstanceNorm3d,
        {"eps": 1e-05, "affine": True},
        None,
        None,
        nn.LeakyReLU,
        {'inplace': True},
        True,
    )
    print('Fully residual Unet Basic Block: ', end="")
    print("{:,}".format(res_unet.compute_conv_feature_map_size(input_shape[2:])))
    # 7,896,932,352

    res_unet_bottleneck = ResidualUNet(
        in_channels,
        n_stages,
        features_per_stage,
        conv_op,
        kernel_sizes,
        initial_strides,
        n_blocks_per_stage,
        2,
        n_conv_per_stage_decoder,
        True,
        nn.InstanceNorm3d,
        {"eps": 1e-05, "affine": True},
        None,
        None,
        nn.LeakyReLU,
        {'inplace': True},
        True,
        block=BottleneckD,
        bottleneck_channels=[f * 4 for f in features_per_stage]
    )
    print('Fully residual Unet Bottleneck Block: ', end="")
    print("{:,}".format(res_unet_bottleneck.compute_conv_feature_map_size(input_shape[2:])))
    # 22,873,808,896

    mednext = MedNeXT(
        in_channels,
        n_stages,
        features_per_stage,
        4,
        conv_op,
        3,
        initial_strides,
        True,
        n_blocks_per_stage,
        2,
        n_conv_per_stage_decoder,
        nn.GroupNorm,
        None,
        nn.GELU,
        {},
        True
    )
    print('MedNeXT: ', end="")
    print("{:,}".format(mednext.compute_conv_feature_map_size(input_shape[2:])))
    # 22,457,720,832

if __name__ == "__main__":
    main()