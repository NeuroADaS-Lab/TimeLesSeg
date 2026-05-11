import numpy as np

def zscore_norm(data: np.ndarray, use_mask_for_norm: bool, seg: np.ndarray = None) -> np.ndarray:
    if use_mask_for_norm and seg is not None:
        mask = seg >= 0
        mean = data[mask].mean()
        std = data[mask].std()
        data[mask] = (data[mask] - mean) / (max(std, 1e-8))
    else:
        mean = data.mean()
        std = data.std()
        data -= mean
        data /= (max(std, 1e-8))
    return data

def no_normalization(data: np.ndarray, use_mask_for_norm: bool, seg: np.ndarray = None) -> np.ndarray:
    return data

if __name__ == "__main__":
    import sys
    from timelesseg.data_generation.image import load_nifti
    # import timeit
    data = load_nifti(sys.argv[1])
    _data = data.data.astype(np.float32)
    seg = load_nifti(sys.argv[2])
    _seg = seg.data

    # print('With seg:')
    # print('With np.ones')
    # print(timeit.timeit(lambda: zscore_norm(_data, _seg), number=1000))
    # print('Without np.ones')
    # print(timeit.timeit(lambda: _zscore_norm(_data, _seg), number=1000))

    # print()
    # print('Without seg:')
    # print('With np.ones')
    # print(timeit.timeit(lambda: zscore_norm(_data, None), number=1000))
    # print('Without np.ones')
    # print(timeit.timeit(lambda: _zscore_norm(_data, None), number=1000))