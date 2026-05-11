import multiprocessing
import numpy as np
from scipy.ndimage import label

from timelesseg.utils import save_json
from timelesseg.io import rw
from timelesseg.dataloading import generate_iterable_with_fnames

from .json_export import recursive_fix_for_json_export
from . import dice as dc

## METRICS USED TO EVALUATE TRAINING

def compute_metrics(gt_file: str, prediction_file: str, baseline_file: str) -> dict:
    # load images
    gt, gt_properties = rw.read_seg(gt_file)
    pred, pred_properties = rw.read_seg(prediction_file)
    assert gt.shape == pred.shape, gt_file

    # I don't have regions, classes, etc.
    # So for the moment keep it simple.
    gt = gt == 1
    pred = pred == 1

    baseline = baseline_file
    if baseline_file is not None:
        baseline, _ = rw.read_seg(baseline_file)
        assert gt.shape == baseline.shape, baseline_file
        baseline = baseline == 1

    results = {}
    results['gt_file'] = gt_file
    results['prediction_file'] = prediction_file
    results['baseline_file'] = baseline_file

    results['metrics'] = {}

    tp, fp, fn, tn = dc.compute_tp_fp_fn_tn(gt, pred, baseline)
    union = tp + fp + fn
    if union == 0:
        results['metrics']['Dice'] = results['metrics']['IoU'] = np.nan
    else:
        num_dice = 2 * tp
        results['metrics']['Dice'] = num_dice / (num_dice + fp + fn)
        results['metrics']['IoU'] = tp / union

    results['metrics']['FP'] = fp
    results['metrics']['TP'] = tp
    results['metrics']['FN'] = fn
    results['metrics']['TN'] = tn
    results['metrics']['n_pred'] = fp + tp
    results['metrics']['n_ref'] = fn + tp

    return results


def compute_metrics_on_folder(gt_folder: str,
                              dict_with_preds: dict[str, str],
                              output_file: str | None,
                              num_processes: int):
    """
    output_file must end with .json; can be None
    """
    iterable_gts = generate_iterable_with_fnames(gt_folder, allow_no_seg=False, file_ending = '.nii.gz')
    identifiers = set(iterable_gts.keys())
    assert identifiers == set(dict_with_preds.keys())

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            [
                (iterable_gts[identifier]['seg'], dict_with_preds[identifier], iterable_gts[identifier]['images'][-1])
                for identifier in identifiers
            ] +
            [
                (iterable_gts[identifier]['seg'], dict_with_preds[identifier], None)
                for identifier in identifiers
            ]
        )

    metric_list = list(results[0]['metrics'].keys())
    # mean metric per class
    results_xor, results_regular = results[:len(identifiers)], results[len(identifiers):]
    means_regular = {}
    means_xor = {}
    for m in metric_list:
        means_xor[m] = np.nanmean([i['metrics'][m] for i in results_xor])
        means_regular[m] = np.nanmean([i['metrics'][m] for i in results_regular])

    result = {'metric_per_case': results, 'means_xor': means_xor, 'means_regular': means_regular}
    recursive_fix_for_json_export(result)

    if output_file is not None:
        save_json(result, output_file)

    return result

if __name__ == "__main__":
    folder = 'validation_data/raw'
    dict_with_preds = {k: v['seg'] for k, v in generate_iterable_with_fnames(folder).items()}
    compute_metrics_on_folder(gt_folder=folder,
                              dict_with_preds=dict_with_preds,
                              output_file='test.json', num_processes=16)