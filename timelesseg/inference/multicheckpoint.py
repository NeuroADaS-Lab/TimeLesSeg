import os
import subprocess
import torch
import numpy as np
from typing import Iterable

from timelesseg.utils import write_pickle
from timelesseg.io import rw
from timelesseg.trained_models import CHECKPOINTS
from timelesseg.config import get_preprocessing_config_from_dataset_fingerprint, DATASET_FINGERPRINT_PATH

from .predictor import Predictor, preprocessing_iterator_fromfiles, get_default_device
from .logits_to_probabilities import convert_probabilities_to_segmentation, apply_nonlinearity_to_logits
from .export_prediction import (
    PrepConfig,
    resample_probabilities,
    revert_cropping_on_probabilities
)


class MultiCheckpointPredictor:

    def __init__(self,
                 *checkpoints: str,
                 **predictor_kwargs):

        # default checkpoint paths
        if not len(checkpoints):
            checkpoints = (
                CHECKPOINTS.get('best'),
                CHECKPOINTS.get('final')
            )

        self.predictors = self.initialize_predictors(checkpoints, **predictor_kwargs)
        self.preprocessing_kwargs = self.predictors[0]._get_preprocessing_kwargs()
        self.preprocessing_config = predictor_kwargs.get('preprocessing_config')
        self.device = predictor_kwargs.get('device', get_default_device())

    @classmethod
    def from_checkpoint_path(
        cls, checkpoint_path: Iterable[str], **predictor_kwargs
    ):
        return cls(*checkpoint_path, **predictor_kwargs)

    @staticmethod
    def initialize_predictors(checkpoints: tuple[str],
                              preprocessing_config,
                              device,
                              verbose: bool = True,
                              verbose_preprocessing: bool = False,
                              allow_tqdm: bool = True,
                              masked_normalization: bool = False) -> list[Predictor]:
        ret = []
        for checkpoint in checkpoints:
            # this mimicks inference_entrypoint
            ret.append(
                Predictor.from_checkpoint_path(checkpoint, preprocessing_config,
                                               allow_tqdm=allow_tqdm,
                                               verbose=verbose,
                                               verbose_preprocessing=verbose_preprocessing,
                                               device=device,
                                               masked_normalization=masked_normalization)
            )
        return ret


    def predict_from_list_of_case_dicts(self,
                                        list_of_case_dicts: list[dict],
                                        outfiles: list[str] | None,
                                        num_processes_prep: int,
                                        num_threads_torch: int = 8,
                                        save_probabilities: bool = False) -> list:

        data_iterator = preprocessing_iterator_fromfiles(
            list_of_case_dicts,
            outfiles,
            self.preprocessing_kwargs,
            num_processes_prep,
            pin_memory=self.device == "cuda"
        )

        ret = []
        for preprocessed_item in data_iterator:
            identifier = preprocessed_item['identifier']
            print(f"Processing case: {identifier}")

            case_properties = preprocessed_item['data_properties']
            outfile = preprocessed_item['ofile']

            results_for_this_case = []
            for predictor in self.predictors:
                logits = predictor.predict_single_item(preprocessed_item)
                probabilities = convert_logits_to_probabilities(logits,
                                                                case_properties,
                                                                self.preprocessing_config,
                                                                num_threads_torch=num_threads_torch)
                results_for_this_case.append(probabilities)

            probability_avg = np.mean(results_for_this_case, axis=0)

            if outfile is not None:
                export_prediction_from_probabilities(probability_avg,
                                                     case_properties,
                                                     save_probabilities,
                                                     outfile)
                ret.append(outfile)
            else:
                _ret = (probability_avg, case_properties)
                ret.append(_ret)

            print(f"Done with {identifier}")

        return ret


def convert_logits_to_probabilities(
    predicted_logits: torch.Tensor | np.ndarray,
    properties: dict,
    preprocessing_config: PrepConfig,
    num_threads_torch: int
):
    """
    Here we have to revert the operations performed during preprocessing, in inverse order.

    The operations are:
    3 - Resampling
    2 - Cropping
    1 - Transposing
    (normalization omitted for obvious reasons)
    """

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    ## Revert resampling (resample from target_spacing to original spacing)
    # Since properties['spacing'] is "untransposed", first transpose it
    spacing_transposed = [properties['spacing'][i] for i in preprocessing_config.transpose_forward]
    current_spacing = preprocessing_config.target_spacing
    predicted_logits = resample_probabilities(
        predicted_logits,
        properties['shape_after_cropping'],
        current_spacing,
        spacing_transposed
    )

    predicted_probabilities = apply_nonlinearity_to_logits(predicted_logits).numpy()
    predicted_probabilities = revert_cropping_on_probabilities(predicted_probabilities,
                                                               properties['bbox_used_for_cropping'],
                                                               properties['shape_before_cropping'])

    # Revert transpose on probabilities
    predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in preprocessing_config.transpose_backward])
    torch.set_num_threads(old_threads)
    return predicted_probabilities


def export_prediction_from_probabilities(
    probabilities_array: np.ndarray | torch.Tensor,
    properties: dict,
    save_probabilities: bool,
    outfile: str,
    file_ending: str = '.nii.gz'
):
    """
    :param outfile: has to be truncated! Not including the extension (i.e., `.nii.gz`)
    """
    segmentation = convert_probabilities_to_segmentation(probabilities_array)
    if save_probabilities:
        np.savez_compressed(outfile + '.npz', probabilities=probabilities_array)
        write_pickle(properties, outfile + '.pkl')
        del probabilities_array
    rw.write_seg(segmentation, outfile + file_ending, properties)


def initialize_predictor(checkpoint_paths: Iterable[str], verbose: bool, use_masked_norm: bool, device):
    prep_config = get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH)
    predictor = MultiCheckpointPredictor.from_checkpoint_path(checkpoint_paths,
                                                              preprocessing_config=prep_config,
                                                              allow_tqdm=False,
                                                              verbose=verbose,
                                                              verbose_preprocessing=verbose,
                                                              device=device,
                                                              masked_normalization=use_masked_norm)
    return predictor

def inference_from_list_of_case_dicts(list_of_case_dicts: list[dict],
                                      output_files: list[str] | None,
                                      checkpoint_paths: Iterable[str],
                                      verbose: bool,
                                      device,
                                      num_processes_pp: int,
                                      num_processes_export: int,
                                      save_probabilities: bool,
                                      use_masked_norm: bool):

    predictor = initialize_predictor(checkpoint_paths, verbose, use_masked_norm, device)

    return predictor.predict_from_list_of_case_dicts(list_of_case_dicts,
                                                     output_files,
                                                     num_processes_pp,
                                                     num_processes_export,
                                                     save_probabilities)


def get_folder_from_images(images: tuple[str]):
    im, baseline = images
    if baseline is None:
        return 'evaluation/my-method/probabilities_ensemble_checkpoints_best_final/cross_sectional'
    if os.path.relpath(baseline).startswith('test-data'):
        return 'evaluation/my-method/probabilities_ensemble_checkpoints_best_final/longitudinal'
    return 'evaluation/my-method/probabilities_ensemble_checkpoints_best_final/half_longitudinal'


def test():
    case_dicts = [
        {'images': ('test-data/isbi_2015_data/training/training04/preprocessed/training04_01_flair_pp.nii.gz', None)},
        {'images': ('test-data/isbi_2015_data/training/training04/preprocessed/training04_02_flair_pp.nii.gz', 'test-data/isbi_2015_data/training/training04/masks/training04_01_union.nii.gz')},
        {'images': ('test-data/open_ms_data/cross_sectional/coregistered/patient14/T2W.nii.gz', None)}
    ]
    os.makedirs('test-multi-checkpoint', exist_ok=True)
    outfiles = [
        'test-multi-checkpoint/test1',
        'test-multi-checkpoint/test2',
        'test-multi-checkpoint/test3'
    ]
    targets = [
        'evaluation/my-method-global-norm/checkpoint_ensemble_final_best/cross_sectional/training04_1_flair_prediction.nii.gz',
        'evaluation/my-method-global-norm/checkpoint_ensemble_final_best/longitudinal/training04_2_flair_prediction.nii.gz',
        'evaluation/my-method-global-norm/checkpoint_ensemble_final_best/cross_sectional/patient14_None_T2W_prediction.nii.gz'
    ]

    checkpoints = [
        CHECKPOINTS['final'],
        CHECKPOINTS['best']
    ]
    predictor = initialize_predictor(checkpoints, verbose=True, use_masked_norm=False)
    predictor.predict_from_list_of_case_dicts(case_dicts,
                                              outfiles,
                                              3,
                                              save_probabilities=True)
    for o, t in zip(outfiles, targets):
        o = subprocess.check_output(
            ['seg_stats', o + '.nii.gz', '-d', t]
        )
        print(o.decode())


if __name__ == "__main__":
    predictor1 = MultiCheckpointPredictor.from_checkpoint_path([CHECKPOINTS.get('final'), CHECKPOINTS.get('early')],
                                                               preprocessing_config=get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH),
                                                               verbose=True)
    assert len(predictor1.predictors) == 2
    del predictor1

    predictor1 = MultiCheckpointPredictor.from_checkpoint_path([CHECKPOINTS.get('final'), CHECKPOINTS.get('early'), CHECKPOINTS.get('best')],
                                                               preprocessing_config=get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH),
                                                               verbose=True)
    assert len(predictor1.predictors) == 3

    del predictor1

    predictor1 = MultiCheckpointPredictor.from_checkpoint_path([CHECKPOINTS.get('final'), CHECKPOINTS.get('early'), CHECKPOINTS.get('best'),
                                                               CHECKPOINTS.get('final')],
                                                               preprocessing_config=get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH),
                                                               verbose=True)
    assert len(predictor1.predictors) == 4
    
    test()