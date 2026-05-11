import os
from tqdm import tqdm
from time import sleep
import multiprocessing
from queue import Queue
from threading import Thread
import itertools

import torch
from torch import nn
from torch.amp import autocast
import numpy as np

from timelesseg.utils import get_default_device
from timelesseg.preprocessing.preprocessing import get_normalizers_inference, get_resamplers, preprocess_case
from timelesseg.trainer.utils import empty_cache, dummy_context

from .preprocessing import preprocessing_iterator_fromfiles
from .export_prediction import export_prediction_from_logits
from .utils import (
    pad_nd_image,
    compute_steps_for_sliding_window,
    compute_gaussian,
    check_workers_alive_and_busy
)

from .load_models import load_model


class Predictor:
    """
    :param preprocessing_config: required for preprocessing inputs.
    :param network: to allow easy initialization from Trainer.
    """
    def __init__(
        self,
        network: nn.Module,
        training_config,
        arch_kwargs,
        allowed_mirror_axes: tuple[int, ...],
        preprocessing_config,
        tile_step_size: float,
        use_gaussian: bool,
        use_mirroring: bool,
        masked_normalization: bool,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device(get_default_device()),
        verbose: bool = True,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True
    ):
        """
        Predictor can be initialized either with a path to a checkpoint (easier), or by supplying the network (initialized),
        training_config, arch_kwargs, allowed_mirror_axes

        :param network:
        :param training_config:
        :param arch_kwargs:
        :param allowed_mirror_axes:
        :param preprocessing_config: required for preprocessing, duh
        :param manual_initialization_kwargs: if network_or_path_to_weigths is an nn.Module, this should contain: `training_config, arch_kwargs, allowed_mirror_axes`
        """

        self.preprocessing_config = preprocessing_config
        self.tile_step_size = tile_step_size
        self.verbose = verbose
        self.verbose_pp = verbose_preprocessing
        self.device = device
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.masked_normalization = masked_normalization

        self.perform_everything_on_device = perform_everything_on_device

        self.allow_tqdm = allow_tqdm

        # quick check bcause I'm sick and tired of this happenning to me
        if device.type == "mps" and 'AvgPool3d' in repr(network) and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != "1":
            msg = (
                'PYTORCH_ENABLE_MPS_FALLBACK is not enabled and the network has AvgPool3d in its architecture. '
                'This operation has not been implemented yet. Please re-run the prediction script after setting '
                'PYTORCH_ENABLE_MPS_FALLBACK="1"'
            )
            raise RuntimeError(msg)

        (
            self.network,
            self.training_config,
            self.network_kwargs,
            self.allowed_mirror_axes
        ) = self._initialize_manually(network, training_config, arch_kwargs, allowed_mirror_axes)


    @classmethod
    def from_checkpoint_path(cls,
                             checkpoint_path: str,
                             preprocessing_config,
                             tile_step_size: float = 0.5,
                             use_gaussian: bool = True,
                             use_mirroring: bool = True,
                             masked_normalization: bool = True,
                             perform_everything_on_device: bool = True,
                             device: str | torch.device = get_default_device(),
                             verbose: bool = True,
                             verbose_preprocessing: bool = False,
                             allow_tqdm: bool = True):

        network, training_config, arch_kwargs, allowed_mirror_axes = cls._initialize_from_checkpoint(checkpoint_path)

        if isinstance(device, str):
            device = torch.device(device)

        predictor = cls(
            network, training_config, arch_kwargs, allowed_mirror_axes,
            preprocessing_config, tile_step_size, use_gaussian, use_mirroring,
            masked_normalization, perform_everything_on_device, device, verbose,
            verbose_preprocessing, allow_tqdm
        )
        return predictor


    @staticmethod
    def _initialize_from_checkpoint(
        checkpoint_path: str
    ) -> tuple:

        # see .load_models
        checkpoint: dict = load_model(checkpoint_path, map_location='cpu', weights_only=False)

        arch_kwargs = checkpoint['init_args']['arch_kwargs']
        arch_kwargs.deep_supervision = False

        training_config = checkpoint['init_args']['training_config']

        network: nn.Module = training_config['network_class'](**arch_kwargs)

        network_parameters = checkpoint['network_weights']
        network.load_state_dict(network_parameters)
        allowed_mirror_axes = checkpoint['allowed_mirror_axes']

        return network, training_config, arch_kwargs, allowed_mirror_axes


    def _initialize_manually(
        self,
        network: nn.Module,
        training_config,
        arch_kwargs,
        allowed_mirror_axes
    ):
        network.eval()
        network.decoder.deep_supervision = False
        return network, training_config, arch_kwargs, allowed_mirror_axes


    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> torch.Tensor:

        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

            if self.verbose:
                print(f'Input shape: {input_image.shape}')
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirror_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.training_config.patch_size,
                                                       'constant', {'value': 0}, True, None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device != 'cpu':
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                except RuntimeError:
                    print('Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        return predicted_logits


    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, default_num_processes: int = 8) -> torch.Tensor:
        """
        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape

        :param default_num_processes: set to 8 to mimick nnUNet (https://github.com/MIC-DKFZ/nnUNet/blob/8c4184d46b60059ff7dc8f74cd535e13554bdeca/nnunetv2/configuration.py#L5)
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)

        prediction = self.predict_sliding_window_return_logits(data).to('cpu')

        if self.verbose:
            print('Prediction done')
        torch.set_num_threads(n_threads)

        return prediction


    def _internal_get_sliding_window_slicers(self, image_size) -> list[tuple[slice, ...]]:

        slicers = []

        patch_size = self.training_config.patch_size
        steps = compute_steps_for_sliding_window(image_size, patch_size, self.tile_step_size)

        if self.verbose:
            print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}'
            )

        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple(
                            [
                                slice(None),
                                *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), patch_size)]
                            ]
                        )
                    )

        return slicers


    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(d, slh, q):
            for s in slh:
                q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
            q.put('end')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')

            shape = (self.network_kwargs.num_classes, *data.shape[1:])
            predicted_logits = torch.zeros(shape,
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.training_config.patch_size),
                    sigma_scale= 1./8,
                    value_scaling_factor=10,
                    device=results_device
                )
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item == 'end':
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            # predicted_logits /= n_predictions
            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits


    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirror_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)

        return prediction


    def predict_from_data_iterator(
        self,
        data_iterator,
        num_processes_segmentation_export: int,
        save_probabilities: bool = False
    ):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' (can be None), 'data_properties' and 'identifier' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']

                ofile = preprocessed['ofile']

                print('Predicting case "%s"' % preprocessed['identifier'])

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

                print('Converting prediction to logits and saving.')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_logits,
                        [(prediction, properties, self.preprocessing_config,
                          save_probabilities, ofile, '.nii.gz')]
                    )
                )

                print('Done with case "%s"' % preprocessed['identifier'])

            ret = [i.get()[0] for i in r]

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


    def _get_preprocessing_kwargs(self) -> dict:
        # masked_norm controls whether use_mask_for_norm is used for channel 0
        normalizers_per_channel = get_normalizers_inference(masked_norm=self.masked_normalization)
        resampler_data, resampler_seg = get_resamplers()
        preprocessing_kwargs = {
            'target_spacing': self.preprocessing_config.target_spacing,
            'transpose_forward': self.preprocessing_config.transpose_forward,
            'normalizers_per_channel': normalizers_per_channel,
            'resampling_function': resampler_data,
            'resampling_seg_function': resampler_seg, # this won't get used, but to be consistent...
            'verbose': self.verbose_pp
        }
        return preprocessing_kwargs

    def predict_single_item(self, preprocessed_item: dict) -> np.ndarray:
        """
        Parameters
        -----
        :preprocessed_item: as returned by preprocess_fromfiles_save_to_queue/preprocessing_iterator_fromfiles
        """
        data = preprocessed_item['data']
        with torch.no_grad():
             prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()
        return prediction


    def predict_from_list_of_case_dicts(
        self,
        list_of_case_dicts: list[dict],
        outfiles: list[str],
        num_processes_prep: int,
        num_processes_export: int,
        save_probabilities: bool = False
    ):
        preprocessing_kwargs = self._get_preprocessing_kwargs()
        data_iterator = preprocessing_iterator_fromfiles(list_of_case_dicts, outfiles,
                                                         preprocessing_kwargs,
                                                         num_processes_prep,
                                                         pin_memory=self.device == "cuda")

        return self.predict_from_data_iterator(data_iterator, num_processes_export, save_probabilities=save_probabilities)


    def predict_from_list_of_case_dicts_sequential(
        self,
        list_of_case_dicts: list[dict],
        outfiles: list[str] | None,
        save_probabilities: bool
    ):
        preprocessing_kwargs = self._get_preprocessing_kwargs()

        for i, case_dict in enumerate(list_of_case_dicts):
            data, seg, data_properties = preprocess_case(case_dict['images'],
                                                         seg_path=None,
                                                         preprocessing_kwargs=preprocessing_kwargs)
            del seg

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            prediction = self.predict_logits_from_preprocessed_data(data).cpu()
            export_prediction_from_logits(prediction, data_properties,
                                          self.preprocessing_config, save_probabilities,
                                          outfiles[i])

        compute_gaussian.cache_clear()
        empty_cache(self.device)

        return
