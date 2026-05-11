import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def naive_ce(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param x: input logits
    :param y: one-hot encoded target
    """
    if y.ndim != x.ndim:
        y = y.view(y.shape[0], 1, *x.shape[2:])

    num_p = torch.exp(x)
    denom_p = num_p.sum(dim=1, keepdim=True)
    probs = num_p / denom_p

    if y.shape != x.shape:
        y_onehot = torch.zeros_like(x)
        y_onehot.scatter_(1, y.long(), 1)
    else:
        y_onehot = y

    return - torch.log((probs * y_onehot).sum(1))


def test_ce_losses():
    ce_loss = CrossEntropyLoss(
        reduction='none'
    )

    BATCH_SIZE = 4
    NCLASSES = 5
    SPATIAL_SIZE = (256, 182, 196)
    gt = torch.randint(0, NCLASSES, size=(BATCH_SIZE, *SPATIAL_SIZE))
    prediction = torch.from_numpy(
        np.random.uniform(-5, 5, size=(BATCH_SIZE, NCLASSES, *SPATIAL_SIZE))
    )

    L_true = ce_loss(prediction, gt)
    L_got = naive_ce(prediction, gt)
    assert torch.allclose(L_true, L_got)
    

if __name__ == "__main__":
    test_ce_losses()