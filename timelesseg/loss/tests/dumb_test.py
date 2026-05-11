import torch

NCLASSES = 5
def get_seg():
    return torch.randint(low=0, high=NCLASSES, size=(5, 128, 128, 128))

def to_onehot(seg: torch.Tensor):
    seg = seg.view((seg.shape[0], 1, *seg.shape[1:]))
    y_onehot = torch.zeros((seg.shape[0], NCLASSES, *seg.shape[2:]), device=seg.device, dtype=torch.bool)
    y_onehot.scatter_(1, seg.long(), 1)
    return y_onehot

def test():
    seg = get_seg()
    seg_onehot = to_onehot(seg)
    assert torch.equal(torch.argmax(seg_onehot.to(torch.int8), dim=1), seg)
    batch = False
    reduce_axis: list[int] = torch.arange(2, seg_onehot.ndim).tolist()
    if batch:
        raise RuntimeError
        reduce_axis = [0] + reduce_axis

    ground_o = torch.sum(seg_onehot, reduce_axis)
    w_func = lambda grnd: torch.reciprocal(grnd * grnd)
    w = w_func(ground_o.float())
    infs = torch.isinf(w)
    if batch:
        raise RuntimeError
        w[infs] = 0.0
        w = w + infs * torch.max(w)
    else:
        w[infs] = 0.0
        print(w.shape)
        print(torch.max(w, dim=1))
        max_values = torch.max(w, dim=1)[0].unsqueeze(dim=1)
        print(max_values.shape)
        w = w + infs * max_values

if __name__ == "__main__":
    test()