from time import sleep
import multiprocessing

import torch
from torch import nn

from timelesseg.utils import join, maybe_mkdir
from timelesseg.configs.preprocessing import get_preprocessing_config_from_dataset_fingerprint
from timelesseg.dataloading import Dataset
from timelesseg.inference.predictor import Predictor, compute_gaussian
from timelesseg.inference.utils import check_workers_alive_and_busy
from timelesseg.inference.export_prediction import export_prediction_from_logits


def final_validation_from_trainer(
    network: nn.Module | str,
    device: str | torch.device,
    training_config,
    arch_kwargs,
    allowed_mirror_axes: tuple[int, ...],
    validation_results_folder: str,
    preprocessing_config = None
) -> dict[str, str]:

    if preprocessing_config is None:
        preprocessing_config = get_preprocessing_config_from_dataset_fingerprint(join(training_config.training_data_path, 'dataset_fingerprint.json'))

    if isinstance(device, str):
        device = torch.device(device)

    predictor = Predictor(network, training_config, arch_kwargs,
                          allowed_mirror_axes, preprocessing_config,
                          tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                          perform_everything_on_device=True, device=device, verbose=True,
                          verbose_preprocessing=True, allow_tqdm=True)

    validation_dataset = Dataset(folder=training_config.validation_data_path)

    return _final_validation(predictor, validation_dataset, training_config.num_processes, validation_results_folder, preprocessing_config)


def final_validation(predictor, validation_dataset_path_or_dataset: str | Dataset, num_processes, validation_results_folder, preprocessing_config):
    if isinstance(validation_dataset_path_or_dataset, str):
        validation_dataset_path_or_dataset = Dataset(validation_dataset_path_or_dataset)
    return _final_validation(predictor, validation_dataset_path_or_dataset, num_processes, validation_results_folder, preprocessing_config)


def _final_validation(
    predictor: Predictor,
    validation_dataset: Dataset,
    num_processes: int,
    validation_results_folder: str,
    preprocessing_config
) -> dict[str, str]:

    maybe_mkdir(validation_results_folder)
    dict_with_preds = {}
    with multiprocessing.get_context("spawn").Pool(num_processes) as segmentation_export_pool:
        worker_list = [i for i in segmentation_export_pool._pool]
        results = []

        for identifier in validation_dataset.identifiers:
            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                       allowed_num_queued=2)
            while not proceed:
                sleep(0.1)
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                            allowed_num_queued=2)

            data, seg, properties = validation_dataset.load_case(identifier)

            data = torch.from_numpy(data)

            output_file = join(validation_results_folder, 'case_%s.nii.gz' % identifier)

            prediction = predictor.predict_sliding_window_return_logits(data).cpu()

            results.append(
                segmentation_export_pool.starmap_async(
                    export_prediction_from_logits, [
                        (prediction, properties, preprocessing_config,
                            False, output_file, '.nii.gz', num_processes),
                    ]
                )
            )

            dict_with_preds[identifier] = output_file

        _ = [r.get()[0] for r in results]

    compute_gaussian.cache_clear()
    return dict_with_preds