from torch import nn
if False:
    from MedNeXt_original.nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
from timelesseg.models.utils import check_two_nets_are_equal, features_per_stage
from timelesseg.models.MedNeXT.mednextv2 import MedNeXT


def compare_original_and_mine():
    in_channels = 1
    num_stages = 5
    n_classes = 2
    feats = features_per_stage(32, num_stages, 99999)
    expansion_ratio = [2] * (num_stages + (num_stages-1))
    ks = [3] * (num_stages)
    n_convs_per_stage_encoder = [2] * num_stages
    n_convs_per_stage_decoder = [2] * (num_stages - 1)

    original = MedNeXt(in_channels,
                       feats[0],
                       n_classes,
                       expansion_ratio, kernel_size=3, deep_supervision=True,
                       do_res=True, do_res_up_down=True, block_counts=n_convs_per_stage_encoder + n_convs_per_stage_decoder, checkpoint_style=None,
                       grn=True, dim='3d')
    # print(original)
    # exit()

    mine = MedNeXT(
        in_channels,
        num_stages,
        feats,
        expansion_ratio,
        nn.Conv3d,
        num_classes=n_classes,
        kernel_sizes=ks, strides = [1] + [2] * (num_stages-1),
        n_conv_per_stage=n_convs_per_stage_encoder,
        n_conv_per_stage_decoder=n_convs_per_stage_decoder, deep_supervision=True
    )
    print(mine)

    # assert check_two_nets_are_equal(original, mine)

if __name__ == "__main__":
    compare_original_and_mine()
