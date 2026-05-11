import numpy as np
from scipy.ndimage import label, generate_binary_structure
import pandas as pd # TODO use pd.unique(seg.ravel())


def compute_tp_fp_fn_tn(
    gt: np.ndarray,
    pred: np.ndarray,
    baseline: np.ndarray = None
) -> tuple[int, int, int, int]:

    if baseline is not None:
        gt ^= baseline
        pred ^= baseline

    # true positives:
    # voxels that are lesion and
    # have been predicted as such
    tp = np.sum(gt & pred)

    # false positives:
    # voxels that are not lesion (positive)
    # BUT have been predicted as such
    fp = np.sum(~gt & pred)

    # false negatives:
    # voxels that are actually lesion
    # but have not been predicted correctly
    fn = np.sum(gt & ~pred)

    # true negatives:
    # voxels that are "healthy"
    # and are predicted as such
    tn = np.sum(~gt & ~pred)

    return tp, fp, fn, tn

def _dice(tp, fp, fn, tn) -> float:
    num = 2 * tp
    denom = num + fp + fn
    if denom == 0:
        return np.nan
    return num / denom

def _dice_from_IoU(iou):
    if np.isnan(iou):
        return iou
    return 2 * iou / (iou + 1)

def _IoU(tp, fp, fn, tn) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return np.nan
    return tp / denom

def _dice_and_IoU(tp, fp, fn, tn):
    iou = _IoU(tp, fp, fn, tn)
    dc = _dice_from_IoU(iou)
    return dc, iou

def dice(gt: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _dice(tp, fp, fn, tn)

def IoU(gt: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _IoU(tp, fp, fn, tn)

def dice_and_IoU(gt, pred) -> tuple[float, float]:
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _dice_and_IoU(tp, fp, fn, tn)

## lesional true positive rate
## see https://doi.org/10.1016/j.neuroimage.2016.12.064 -- section 2.2
def lesional_tpr_naive(gt: np.ndarray, pred: np.ndarray):
    connected_comp_gt, n_lesions = label(gt)
    num = 0
    denom = 0
    matching_lesions = 0
    for l in range(1, n_lesions+1):
        this_gt = connected_comp_gt == l
        matching_lesion = bool((this_gt * pred).any())

        num += matching_lesion
        matching_lesions += matching_lesion
        denom += 1
    try:
        lesional_dice = num / denom
    except ZeroDivisionError:
        lesional_dice = np.nan

    return lesional_dice, matching_lesions

def _lesional_tpr_efficient(labeled_gt: np.ndarray, nlesions_gt: int, pred: np.ndarray) -> float:
    nlesions_intersection = len(pd.unique((labeled_gt * pred).ravel())[1:])
    return nlesions_intersection / nlesions_gt

def _lesional_tpr_from_tp_and_nlesions(lesional_tp, nlesions_gt):
    return lesional_tp / nlesions_gt

def _lesional_sensitivity(lesional_tp, lesional_fn):
    # TP/(TP + FN)
    return lesional_tp / (lesional_tp + lesional_fn)

def _lesional_fdr(lesional_tp, lesional_fp):
    # FP/(FP + TP))
    return lesional_fp / (lesional_fp + lesional_tp)

def lesional_tpr_efficient(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_labeled, nlesions_gt = label(gt)
    if nlesions_gt == 0:
        return np.nan
    nlesions_intersection = len(np.unique(gt_labeled * pred.astype(bool))[1:])
    return nlesions_intersection / nlesions_gt

def _conformity_coefficient(tp, fp, fn, tn):
    """
    https://doi.org/10.1016/j.neuroimage.2009.03.068
    """
    if tp == 0:
        return np.nan
    return (1 - ((fp + fn)/tp)) * 100

def conformity_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _conformity_coefficient(tp, fp, fn, tn)

def _ppv(tp, fp, fn, tn) -> float:
    denom = tp + fp
    if denom == 0:
        return np.nan
    return tp / denom

def ppv(gt, pred) -> float:
    """Positive Predictive Value"""
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _ppv(tp, fp, fn, tn)

def _tpr(tp, fp, fn, tn):
    denom = tp + fn
    if denom == 0:
        return np.nan
    return tp / denom

def tpr(gt, pred) -> float:
    """True Positive Rate"""
    tp, fp, fn, tn = compute_tp_fp_fn_tn(gt, pred)
    return _tpr(tp, fp, fn, tn)

def _lesional_dice(gt_labeled: np.ndarray,
                   pred_bin: np.ndarray,
                   ncomp_gt: int,
                   ncomp_pred: int,
                   gt_bin: np.ndarray = None):

    if gt_bin is None:
        gt_bin = gt_labeled.astype(bool, copy=True)

    tp = len(pd.unique((gt_labeled * pred_bin).ravel())[1:])
    fp = ncomp_pred - tp
    fn = ncomp_gt - tp

    num = 2 * tp
    denom = num + fp + fn
    if denom == 0:
        return np.nan
    return num / denom, tp, fp, fn

def lesional_dice(gt: np.ndarray, pred: np.ndarray, struct: np.ndarray = None):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    gt_labeled, ncomp_gt = label(gt, structure=struct)
    _, ncomp_pred = label(pred, structure=struct)
    del _

    return _lesional_dice(gt_labeled, pred, ncomp_gt, ncomp_pred, gt)

def _new_lesional_dice(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_labeled: np.ndarray,
    pred_labeled: np.ndarray,
    ncomp_gt: int,
    ncomp_pred: int
):
    all_vals_gt = set(range(1, ncomp_gt+1))
    all_vals_pred = set(range(1, ncomp_pred+1))

    tp_gt = set(pd.unique((gt_labeled * pred).ravel())) - {0}
    tp_pred = set(pd.unique((pred_labeled * gt).ravel())) - {0}

    fp = all_vals_pred - tp_pred
    fn = all_vals_gt - tp_gt
    tp = len(tp_gt)
    fp = len(fp)
    fn = len(fn)
    num = 2 * tp
    denom = num + fp + fn
    dc =  num / denom
    return dc, tp, fp, fn

def new_lesional_dice(
    gt,
    pred
):
    """
    Equivalent to:
    https://github.com/ander-elkoroaristizabal/nnunet-ms-segmentation/blob/44e95ca5f6860a25074ef910a9e91c47464fc660/custom_scripts/F_evaluate_quantitative.py#L104

    See: https://chatgpt.com/share/68f24e1d-75e8-8001-a7fb-da2905c21f1f
    Also: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255939#sec002
    """
    gt_labeled, ncomp_gt = label(gt)
    pred_labeled, ncomp_pred = label(pred)
    return _new_lesional_dice(gt, pred, gt_labeled, pred_labeled, ncomp_gt, ncomp_pred)


if __name__ == "__main__":
    import sys
    import nibabel as nib
    gt = np.asanyarray(nib.load(sys.argv[1]).dataobj).astype(bool)
    pred = np.asanyarray(nib.load(sys.argv[2]).dataobj).astype(bool)

    print(lesional_tpr_naive(gt, pred))
    print(lesional_tpr_efficient(gt, pred))
    print(lesional_dice(gt, pred, struct=generate_binary_structure(3, 3)))