import torch
from torch import nn
if False:
    from MedNeXt_original.nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock, MedNeXtDownBlock
from timelesseg.models.utils import check_two_nets_are_equal, features_per_stage
from timelesseg.models.MedNeXT.blocks import MedNeXtBlock as Mine

def main():

    in_channels = 1
    hidden_dim = 32
    r = 4
    ks = 3

    torch.manual_seed(999)
    his = MedNeXtDownBlock(hidden_dim, hidden_dim, r, ks, do_res = True, grn=True)
    torch.manual_seed(999)
    mine = Mine(nn.Conv3d, hidden_dim, hidden_dim, 2, ks, r, nn.GroupNorm, None, None, None, nn.GELU, {}, residual=True, grn=True, perform_second_nonlin=False)
    print(his)
    print()
    print(mine)
    torch.manual_seed(999)
    data = torch.randn((2, 32, 96, 96, 96))
    torch.manual_seed(999)
    x1 = his(data)
    torch.manual_seed(999)
    x2 = mine(data)

    print(torch.abs(x1 - x2).max())


if __name__ == "__main__":
    main()