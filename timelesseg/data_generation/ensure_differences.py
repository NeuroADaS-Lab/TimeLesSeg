from typing import Union, Callable, Tuple
import numpy as np
import scipy.stats as st

from timelesseg.typing import Number, ArrayLike

# abstraction
_difference_function: Callable[
    [
        Tuple[np.ndarray, np.ndarray], # image and seg
        Union[Number, Tuple[Number, ...]], # thresholds or decision criteria
        Tuple[int, ...], # classes we are interested int
    ],
    bool
]

# def difference_function(
#         *data,
#         classes: Tuple[int, ...] = None,
#         _f: Callable,
#         **_f_kwargs
#     ) -> bool:

#     if len(data) == 1:
#         assert classes is not None
#         data = data[0]
#         original_data = data.copy()
#         _data = []
#         for c in classes:
#             _data.append(data[data == c])
#             assert (data == original_data).all()
#         return difference_function(*_data, classes=None, _f =_f, **_f_kwargs)

#     return _f(*data, **_f_kwargs)


def _ttest_based_criteria(
    data1: np.ndarray,
    data2: np.ndarray,
    threshold: Number,
    equal_var: bool = False,
    permutations: int = None
) -> bool:
    """
    Returns whether the ttest is NOT significant!
    """
    t_stat = st.ttest_ind(data1, data2, equal_var=equal_var, permutations=permutations)
    return t_stat.pvalue >= threshold


def _mean_based_criteria(
    data1: np.ndarray,
    data2: np.ndarray,
    threshold_means: Number
) -> bool:
    """
    Returns whether the difference of means is LOWER than threhold.
    """
    diff_of_means = np.abs(data1.mean() - data2.mean())
    return diff_of_means < threshold_means


def mean_based_criteria(
    wm_array: np.ndarray,
    lesions_array: np.ndarray,
    adaptive: bool,
    threshold_means: Number
) -> bool:
    """
    :param adaptive: compute threshold automatically as the minimum between mean +/- 2 * std of WM and threshold means
    :param threshold_means:
    """
    if adaptive:
        difference_in_wm = wm_array.mean() + wm_array.std() * (np.array([+2, -2]))
        difference_in_wm = np.subtract(*difference_in_wm)
        threshold_means = min(difference_in_wm, threshold_means)
    return _mean_based_criteria(wm_array, lesions_array, threshold_means)


def ttest_based_criteria(
    im: np.ndarray,
    seg: np.ndarray,
    wm_class: int,
    lesions_class: int,
    pval_threshold: float,
    equal_var: bool = False,
    mean_criteria: bool = False,
    adaptive_mean: bool = False,
    mean_threshold: Number = None,
    permutations: int = None
) -> bool:

    # get white matter and lesion values
    wm_vals = im[seg == wm_class]
    les_vals = im[seg == lesions_class]

    ttest_result = _ttest_based_criteria(wm_vals, les_vals, pval_threshold, equal_var, permutations)
    if mean_criteria:
        assert mean_threshold is not None
        mean_result = mean_based_criteria(wm_vals, les_vals, adaptive_mean, mean_threshold)
        # an image is not valid (True) if either the ttest is not significant, or if the means are not sufficiently different
        ttest_result = ttest_result or mean_result
    return ttest_result


def percentile_based_criteria(
    im: np.ndarray,
    seg: np.ndarray,
    wm_class: int,
    lesions_class: int,
    percentile: Number,
    class_vals: ArrayLike
) -> bool:

    means = []
    of_interest = []
    for c in class_vals:
        this_area_mean = im[seg == c].mean()
        if c in [lesions_class, wm_class]:
            of_interest.append(this_area_mean)
        means.append(this_area_mean)

    # sort means in descending order
    means = sorted(means, reverse=True)
    means = [means[i] - means[i+1] for i in range(len(means) - 1)]

    pp = np.percentile(means, percentile)

    # print(f'{of_interest=}')

    return np.abs(np.subtract(*of_interest)) < pp


def percentile_based_criteria_with_std(
    im: np.ndarray,
    seg: np.ndarray,
    wm_class: int,
    lesions_class: int,
    percentile: Number,
    class_vals: ArrayLike
) -> bool:

    means_and_stds = []
    of_interest = []
    for c in class_vals:
        this_area = im[seg == c]
        this_area_mean = this_area.mean()
        this_area_std = this_area.std()
        if c in [wm_class, lesions_class]:
            of_interest.append((this_area_mean, this_area_std))
        means_and_stds.append((this_area_mean, this_area_std))

    # sort means and std according to means
    means_and_stds = sorted(means_and_stds, reverse=True, key = lambda x: x[0])
    differences = [(means_and_stds[i][0] - means_and_stds[i+1][0]) / (means_and_stds[i][1] + means_and_stds[i+1][1]) for i in range(len(means_and_stds) - 1)]

    pp = np.percentile(differences, percentile)

    of_interest_difference = np.abs(of_interest[0][0] - of_interest[1][0]) / (of_interest[0][1] + of_interest[1][1])

    return of_interest_difference < pp

def get_offset(i: int, j: int, N: int) -> int:
    """
    Compute offset from a upper triangle matrix without diagonal.
    """
    assert N >= j > i
    maxcol = N - 1
    initial_offset = np.arange(maxcol, maxcol - i, -1).sum()
    # -1 since python indexing goes from 0 to N-1
    offset = (j - i) - 1
    return initial_offset + offset

def get_upper_triangle_without_diagonal(matrix: np.ndarray):
    """
    Helper to get the upper triangular matrix resulting from the application of a function
    over a list of values to get their pairwise differences.

    For example:
    ```python
    means = np.array([5, 15, 2, 412, 52, 321, 23, 12])
    # difference_of_means contains at the ith, jth element the absolute difference between the ith and the jth elements of means
    difference_of_means = np.abs(means[:, None] - means)
    # now vectorized contains the pairwise differences
    vectorized = get_upper_triangle_without_diagonal(difference_of_means)
    ```
    """
    return matrix[np.triu_indices(matrix.shape[0], k=1)]

def percentile_based_criteria_with_std_no_sort(
    im: np.ndarray,
    seg: np.ndarray,
    wm_class: int,
    lesions_class: int,
    percentile: Number,
    class_vals: ArrayLike
) -> bool:

    means = []
    stds = []
    of_interest_indices = []
    for i, c in enumerate(class_vals):
        this_area = im[seg == c]
        means.append(this_area.mean())
        stds.append(this_area.std())

        if c in [wm_class, lesions_class]:
            of_interest_indices.append(i)

    means = np.array(means)
    stds = np.array(stds)

    difference_of_means = means[:, None] - means
    sum_of_stds = stds[:, None] + stds

    effect_sizes = np.abs(difference_of_means) / sum_of_stds
    effect_sizes = get_upper_triangle_without_diagonal(effect_sizes)

    ith_percentile_effect_size = np.percentile(effect_sizes, percentile)

    index_of_interest = get_offset(*of_interest_indices, N=len(class_vals))
    effect_size_of_interest = effect_sizes[index_of_interest]

    return effect_size_of_interest < ith_percentile_effect_size


if __name__ == "__main__":
    means = np.array([3, 4, 4, 5])
    of_interest = [1, 2]
    difference_of_means = np.abs(means[:, None] - means)
    vectorized = get_upper_triangle_without_diagonal(difference_of_means)
    offset = get_offset(*of_interest, len(means))
    assert means[of_interest[1]] - means[of_interest[0]] == vectorized[offset]

    means = np.array([5, 15, 2, 412, 52, 321, 23, 12])
    of_interest = [3, 4]
    difference_of_means = np.abs(means[:, None] - means)
    vectorized = get_upper_triangle_without_diagonal(difference_of_means)
    offset = get_offset(*of_interest, len(means))
    got = vectorized[offset]
    shouldof_got = abs(means[of_interest[1]] - means[of_interest[0]])
    assert shouldof_got == got, str(shouldof_got) + ' ' + str(got)


    means = np.random.randn(10)
    # imagine you want to compare the 5th and the 7th
    of_interest = [4, 6]
    difference_of_means = np.abs(means[:, None] - means)
    vectorized = get_upper_triangle_without_diagonal(difference_of_means)
    offset = get_offset(*of_interest, len(means))
    got = vectorized[offset]
    shouldof_got = abs(means[of_interest[1]] - means[of_interest[0]])
    assert shouldof_got == got, str(shouldof_got) + ' ' + str(got)