import numpy as np
import torch

from timelesseg.loss.dice import (
    MemoryEfficientSoftDiceLoss
)

EPS=1e-5

def softmax_helper(x):
    return torch.softmax(x, dim=1)

def test_memory_efficient_soft_dice_loss_weights():
    standard_dice = MemoryEfficientSoftDiceLoss(
        nonlin=softmax_helper,
        batch_dice=False,
        do_bg=False,
        smooth=EPS
    )
    new_dice = MemoryEfficientSoftDicewWeightsLoss(
        nonlin=softmax_helper,
        batch_dice=False,
        do_bg=False,
        smooth=EPS,
        weight_type='uniform'
    )

    BATCH_SIZE = 4
    NCLASSES = 5
    SPATIAL_SIZE = (256, 182, 196)
    gt = torch.randint(0, NCLASSES, size=(BATCH_SIZE, *SPATIAL_SIZE))
    prediction = torch.from_numpy(
        np.random.uniform(-5, 5, size=(BATCH_SIZE, NCLASSES, *SPATIAL_SIZE))
    )

    sd = standard_dice(prediction, gt)
    nw = new_dice(prediction, gt)

    assert torch.allclose(sd, nw)

    standard_dice = MemoryEfficientSoftDiceLoss(
        nonlin=softmax_helper,
        batch_dice=True,
        do_bg=False,
        smooth=EPS
    )
    new_dice = MemoryEfficientSoftDicewWeightsLoss(
        nonlin=softmax_helper,
        batch_dice=True,
        do_bg=False,
        smooth=EPS,
        weight_type='uniform'
    )

    BATCH_SIZE = 4
    NCLASSES = 5
    SPATIAL_SIZE = (256, 182, 196)
    gt = torch.randint(0, NCLASSES, size=(BATCH_SIZE, *SPATIAL_SIZE))
    prediction = torch.from_numpy(
        np.random.uniform(-5, 5, size=(BATCH_SIZE, NCLASSES, *SPATIAL_SIZE))
    )

    sd = standard_dice(prediction, gt)
    nw = new_dice(prediction, gt)

    assert torch.allclose(sd, nw)



if __name__ == "__main__":
    test_memory_efficient_soft_dice_loss_weights()