import numpy as np
from torch import nn


# yanked from nnUNet
class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors = None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])


def get_deep_supervision_scales(strides: list[int | list[int]]) -> list:
    deep_supervision_scales = list(
        list(i) for i in 1 / np.cumprod(np.vstack(strides), axis=0)
    )[:-1]
    return deep_supervision_scales


def handle_deep_supervision_weights(deep_supervision_weights: np.ndarray, n_stages: int, n_deep_supervision_stages: int = None) -> np.ndarray:
    """
    :param n_deep_supervision_stages: by default will be the number of stages - 2 (deep supervision is not run in the deepest two stages)
    """
    n_deep_supervision_stages = n_deep_supervision_stages or n_stages - 2
    # cause get_deep_supervision_scales removes last one
    assert n_deep_supervision_stages <= n_stages - 1
    for i in range(n_deep_supervision_stages, n_stages-1):
        deep_supervision_weights[i] = 0
    deep_supervision_weights = deep_supervision_weights / deep_supervision_weights.sum()
    return deep_supervision_weights


if __name__ == "__main__":
    n_stages = 6
    strides = [[1] * 3] + [[2] * 3] * (n_stages - 1)
    print(
        get_deep_supervision_scales(strides)
    )

    ds_scales = get_deep_supervision_scales(strides)
    ds_weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
    _ds_weights = handle_deep_supervision_weights(ds_weights.copy(), n_stages)
    print(_ds_weights)
    _ds_weights = handle_deep_supervision_weights(ds_weights.copy(), n_stages, 1)
    print(_ds_weights)
    _ds_weights = handle_deep_supervision_weights(ds_weights.copy(), n_stages, 3)
    print(_ds_weights)