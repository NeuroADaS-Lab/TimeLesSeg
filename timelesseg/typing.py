import sys
from typing import Union, Iterable, List, Tuple
import numpy as np

# I hate myself
# if sys.version_info >= (3, 12):
#     type Number = int | float
#     type ArrayLike = Iterable[Number] | np.ndarray
#     type ToIterableInt = int | list[int] | tuple[int, ...]
# else:
Number = Union[int, float]
ArrayLike = Union[Iterable[Number], np.ndarray]
ToIterableInt = Union[int, List[int], Tuple[int, ...]]