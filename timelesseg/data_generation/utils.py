from typing import Dict, Iterable, Union, Tuple
from enum import IntEnum
import numpy as np
import nibabel as nib


def subj_id_corresponds_to(identifier: Union[int, str], M: int, P: int):
    """
    :param M: number of initial FLM runs
    :param P: number of synthetic images generated
    """
    if isinstance(identifier, str):
        identifier = int(identifier.lstrip('0'))

    to_be_divisible_with = M * P

    ubd = identifier + to_be_divisible_with - (identifier % to_be_divisible_with)
    return int(ubd / to_be_divisible_with)

# array processing
def bin_seg(seg: np.ndarray) -> np.ndarray:
    return seg.astype(bool).astype(np.uint8)

def map_seg(seg: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    out = seg.astype(np.uint16 if max(mapping.values()) > 255 else np.uint8, copy=True)
    for oldval, newval in mapping.items():
        out[seg == oldval] = newval
    return out

def find_available_integer(iterable_of_integers: Iterable[int]) -> int:
    minimum_val, max_val = min(iterable_of_integers), max(iterable_of_integers)
    consecutive = set(range(minimum_val, max_val + 1))
    available = consecutive - set(iterable_of_integers)
    if len(available):
        return min(available)
    return max_val + 1

# NIBABEL / NDIMAGE
class Axes(IntEnum):
    SAGGITAL = -0 | 0 # Left -> Right
    CORONAL = -1 | 1 # Posterior -> Anterior
    AXIAL = -2 | 2 # Inferior -> Superior

def slice_along_dim(array: np.ndarray, dim: int, slicer: Union[Tuple[int], slice]) -> np.ndarray:
    if isinstance(slicer, tuple):
        slicer = slice(*slicer)

    if dim == 0:
        return array[slicer, ...]
    elif dim == 1:
        return array[:, slicer, :]
    elif dim == 2:
        return array[..., slicer]
    else:
        raise ValueError(f'{dim = }')

def get_max_and_index_max(iterable) -> tuple:
    max_index = (iterable).index((max_:=max(iterable)))
    return max_, max_index

def get_orient(affine: np.ndarray):
    ras_code = nib.aff2axcodes(affine)
    return tuple(string_to_axes(c) for c in ras_code)

def string_to_axes(x: str) -> Axes:
    if x in ['L', 'R']:
        return Axes.SAGGITAL
    elif x in ['P', 'A']:
        return Axes.CORONAL
    elif x in ['I', 'S']:
        return Axes.AXIAL
    else:
        raise ValueError(f'Invalid code "{x}".')

# TESTS
def test_map_seg():
    seg = np.array([[1, 1, 0],
                    [2, 2, 0,],
                    [3, 4, 0]])
    expected = np.array([[1, 1, 0],
                         [99, 99, 0],
                         [3, 4, 0]])
    assert (map_seg(seg, {2: 99}) == expected).all()

    expected = np.array([[1, 1, 0],
                         [99, 99, 0],
                         [3, 56, 0]])
    assert (map_seg(seg, {2: 99, 4: 56}) == expected).all()

    expected = np.array([[2, 2, 0],
                         [99, 99, 0],
                         [3, 56, 0]])
    assert (map_seg(seg, {1: 2, 2: 99, 4: 56}) == expected).all()


if __name__ == "__main__":

    someintegers = {1, 2, 3}
    shouldget = 4
    assert find_available_integer(someintegers) == shouldget

    someintegers = [1, 2, 3, 5]
    assert find_available_integer(someintegers) == 4

    someintegers = (5, 6, 8, 9, 24)
    assert find_available_integer(someintegers) == 7

    someintegers += (7, )
    assert find_available_integer(someintegers) == 10

    M = 15
    P = 25
    got = subj_id_corresponds_to(659, M, P)
    assert got == 2

    assert subj_id_corresponds_to('02365', M, P) == 7

    print(subj_id_corresponds_to('03076', M, P))