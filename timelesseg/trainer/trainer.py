from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable
import sys
import os
import inspect
from time import time
import numpy as np
import torch
from torch.amp import autocast
from torch import GradScaler

from timelesseg.utils import (
    maybe_mkdir,
    save_json,
    join,
    dirname,
    remove,
    isfile,
    basename,
    setup_loggers,
    timestampify
)
from timelesseg.loss import (
    MemoryEfficientSoftDiceLoss,
    DC_and_CE_loss,
    get_tp_fp_fn_tn
)
from timelesseg.dataloading.dataset import Dataset
from timelesseg.data_augmentation.transforms import get_training_transforms, get_validation_transforms
from timelesseg.data_augmentation.rotation import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from timelesseg.dataloading.dataloader import DataLoader
from timelesseg.evaluation.compute_metrics_after_training import compute_metrics_on_folder

if TYPE_CHECKING:
    from timelesseg.configs import TrainingConfig, ArchKwargs

from .deep_supervision import get_deep_supervision_scales, DeepSupervisionWrapper, handle_deep_supervision_weights
from .multithreaded_augmenter import NonDetMultiThreadedAugmenter
from .utils import (
    empty_cache,
    get_debug_information_from_trainer,
    collate_outputs,
    has_been_unpacked,
    dummy_context
)
from ._training_progress import TrainingProgressTracker
from .final_validation import final_validation_from_trainer
from .plotting import plot_batched_segmentations



class Trainer:
    loss: Callable[..., torch.Tensor]

    def __init__(
        self,
        training_config: TrainingConfig,
        arch_kwargs: ArchKwargs,
        device: str, # TODO move to config?
        output_folder: str
    ):
        """
        heavily -- and that's an euphemism -- inspired by nnUNet's nnUNetTrainer
        """
        self.config = training_config

        self.current_epoch = 0
        self.num_epochs = self.config.num_epochs

        self.device = torch.device(device)

        # TODO: Test GradScaler with MPS
        self.grad_scaler = GradScaler(device) if device == "cuda" else None

        self.output_folder = output_folder

        self.deep_supervision_scales = get_deep_supervision_scales(arch_kwargs.strides) if self.config.deep_supervision else None
        self.network, self.optimizer, self.lr_scheduler = self.build_network(arch_kwargs)
        self.loss = self.build_loss(self.config.deep_supervision, arch_kwargs.n_stages, training_config.n_deep_supervision_stages)

        # for later final prediction...
        self.arch_kwargs = arch_kwargs

        # let's make sure deep supervision is on in the network
        if not self.network.decoder.deep_supervision:
            raise RuntimeError

        self.progress = TrainingProgressTracker()
        # allows one to track if there are improvements in the training (EMA == Estimated Moving Average Dice)
        self._best_ema = None

        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]


    def _setup_logging(self):
        logger_name = __name__
        log_file_name = basename(dirname(dirname(__file__))) + '_' + self.__class__.__name__.lower()
        log_file_name = join(self.output_folder, timestampify(log_file_name) + '.txt')
        self.logger = setup_loggers(logger_name, verbosity='DEBUG', log_file=log_file_name, console_verbosity='DEBUG', return_logger=True)


    def build_loss(self,
                   deep_supervision: bool,
                   n_stages: int,
                   n_deep_supervision_stages: int
                   ):

        dice_kwargs = {
            'batch_dice': False,
            'do_bg': False,
            'smooth': 1e-5
        }
        ce_kwargs = {}
        loss = DC_and_CE_loss(dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)
        if deep_supervision:
            weights = np.array([1 / (2 ** i) for i in range(len(self.deep_supervision_scales))])
            weights = handle_deep_supervision_weights(weights, n_stages, n_deep_supervision_stages)
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


    def build_network(self, arch_kwargs: dict):
        network = self.config.network_class(**arch_kwargs).to(self.device)
        optimizer, lr_scheduler = self._build_optimizer_and_scheduler(network)
        return network, optimizer, lr_scheduler


    def _build_optimizer_and_scheduler(self, network):
        optimizer = self.config.optim_class(
            network.parameters(),
            self.config.initial_learning_rate,
            weight_decay=self.config.weight_decay,
            **self.config.optim_kwargs
        )
        lr_scheduler = self.config.scheduler_class(optimizer, self.config.initial_learning_rate, self.config.num_epochs)
        return optimizer, lr_scheduler


    def _get_datasets(self):
        tr_dataset = Dataset(self.config.training_data_path)
        val_dataset = Dataset(self.config.validation_data_path)

        # this extracts npys from npz files
        if not has_been_unpacked(tr_dataset.folder, tr_dataset.identifiers):
            self.logger.info('Unpacking training dataset. This may take a while...')
            tr_dataset.unpack_dataset(self.config.num_processes, remove_npz=True)

        if not has_been_unpacked(val_dataset.folder, val_dataset.identifiers):
            self.logger.info('Unpacking validation dataset.')
            val_dataset.unpack_dataset(self.config.num_processes, remove_npz=False)

        return tr_dataset, val_dataset


    def get_dataloaders(self, patch_size):

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

        # so that predictor knows which axes to do mirroring on
        self.allowed_mirror_axes = mirror_axes

        training_dataset, validation_dataset = self._get_datasets()

        training_transforms = get_training_transforms(patch_size, rotation_for_DA,
                                                      self.deep_supervision_scales,
                                                      mirror_axes, do_dummy_2d_data_aug,
                                                      self.config.p_cross_sectional)

        validation_transforms = get_validation_transforms(self.deep_supervision_scales)

        # I just work on binary seg
        all_labels = list(range(self.arch_kwargs.num_classes))
        training_dataloader = DataLoader(training_dataset, self.config.batch_size,
                                         initial_patch_size, patch_size, all_labels,
                                         self.config.oversample_fg_probability,
                                         transforms=training_transforms)
        validation_dataloader = DataLoader(validation_dataset, self.config.batch_size,
                                           patch_size, patch_size, all_labels,
                                           self.config.oversample_fg_probability,
                                           transforms=validation_transforms)

        mt_training_dataloader = NonDetMultiThreadedAugmenter(training_dataloader, None, num_processes=self.config.num_processes,
                                                              num_cached=max(6, self.config.num_processes // 2), seeds=None,
                                                              pin_memory=self.device.type in ['cuda', 'mps'], wait_time=0.002)
        mt_validation_dataloader = NonDetMultiThreadedAugmenter(validation_dataloader, None, self.config.num_processes//2,
                                                                num_cached=max(3, self.config.num_processes // 4), seeds=None,
                                                                pin_memory=self.device.type in ["cuda", 'mps'], wait_time=0.002)

        _ = next(mt_training_dataloader)
        _ = next(mt_validation_dataloader)

        return mt_training_dataloader, mt_validation_dataloader


    def on_training_start(self):
        """
        * Gets dataloaders
        * Creates out dir
        * Sets up logging
        * Saves some debugging stuff
        """
        maybe_mkdir(self.output_folder)
        self._setup_logging()

        self.dataloader_train, self.dataloader_val = self.get_dataloaders(self.config.patch_size)

        empty_cache(self.device)

        debug_info = get_debug_information_from_trainer(self)
        save_json(debug_info, join(self.output_folder, 'debug.json'))


    def train_step(self, batch: dict) -> dict:

        data: torch.Tensor = batch['data']
        target: torch.Tensor | list[torch.Tensor] = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # autocast should be implemented with mps now...
        # We try it otherwise fall back to dummy_context() TODO
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}


    def validation_step(self, batch: dict, plot_batch: bool) -> dict:

        data: torch.Tensor = batch['data']
        target: torch.Tensor | list[torch.Tensor] = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)


        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.deep_supervision_scales is not None:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        # these are the dimensions through which the loss in computed
        axes = [0] + list(range(2, output.ndim))

        # no need for softmax (supposedly, argmaxxing logits or softmax output should yield the same result...)
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        if plot_batch:
            plot_batched_segmentations(data, output_seg, batch['keys'], join(self.output_folder, 'epoch_%i' % self.current_epoch))
        del output_seg, data

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        # remove background
        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


    def _on_training_epoch_start(self):
        # the following line is equivalent to on_epoch_start
        self.progress.log('epoch_start_timestamps', time(), self.current_epoch)

        self.network.train()
        self.lr_scheduler.step(self.current_epoch)

        self.logger.debug('Start of training epoch %i/%i', self.current_epoch, self.num_epochs)
        self.logger.debug('Current learning rate: %f', np.round(self.optimizer.param_groups[0]['lr'], decimals=5))

        self.progress.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


    def _on_training_epoch_end(self, train_outputs: list[dict]):
        outputs = collate_outputs(train_outputs)
        loss_here = np.mean(outputs['loss'])

        self.progress.log('train_losses', loss_here, self.current_epoch)


    def _on_validation_epoch_start(self):
        self.logger.debug('Start of validation epoch %i/%i', self.current_epoch, self.num_epochs)
        self.network.eval()


    def _on_validation_epoch_end(self, val_outputs: list[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.progress.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.progress.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.progress.log('val_losses', loss_here, self.current_epoch)

    def finish_epoch(self):

        self.progress.log('epoch_end_timestamps', time(), self.current_epoch)

        self.logger.debug('train_loss: %f', np.round(self.progress['train_losses'][-1], decimals=4))
        self.logger.debug('val_loss: %f', np.round(self.progress['val_losses'][-1], decimals=4))
        self.logger.debug('Pseudo dice: %s', str([np.round(i, decimals=4) for i in self.progress['dice_per_class_or_region'][-1]]))

        epoch_time = np.round(self.progress['epoch_end_timestamps'][-1] - self.progress['epoch_start_timestamps'][-1], decimals=2)
        self.logger.debug('Epoch time: %fs', epoch_time)

        # handling periodic checkpointing
        if (self.current_epoch + 1) % self.config.save_every == 0 and self.current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.progress['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.progress['ema_fg_dice'][-1]
            self.logger.debug('Yayy! New best EMA pseudo Dice: %f', np.round(self._best_ema, decimals=4))
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        self.progress.plot_progress_png(self.output_folder)

        self.current_epoch += 1


    def finish_training(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if isfile(join(self.output_folder, "checkpoint_latest.pth")):
            remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, NonDetMultiThreadedAugmenter):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, NonDetMultiThreadedAugmenter):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.logger.info("Training done!!!")


    def save_checkpoint(self, filename: str) -> None:
        """
        TODO: Revise
        """

        checkpoint = {
            'network_weights': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            'logging': self.progress.get_checkpoint(),
            '_best_ema': self._best_ema,
            'current_epoch': self.current_epoch + 1,
            'init_args': self.my_init_kwargs,
            'trainer_name': self.__class__.__name__,
            'allowed_mirror_axes': self.allowed_mirror_axes,
        }
        torch.save(checkpoint, filename)


    def load_checkpoint(self, filename_or_checkpoint: dict | str) -> None:
        """
        TODO: Revise.
        """

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.config = checkpoint['init_args']['training_config']
        self.current_epoch = checkpoint['current_epoch']
        self.progress.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.allowed_mirror_axes = checkpoint.get('allowed_mirror_axes') or self.allowed_mirror_axes

        self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])


    def run_final_validation(self, preprocessing_config = None):

        validation_results_folder = join(self.output_folder, 'final_validation_results')
        predictions_dict = final_validation_from_trainer(
            self.network,
            self.device,
            self.config,
            self.arch_kwargs,
            self.allowed_mirror_axes,
            validation_results_folder,
            preprocessing_config
        )

        results_of_val = compute_metrics_on_folder(join(dirname(self.config.validation_data_path), 'raw'),
                                                   predictions_dict,
                                                   join(self.output_folder, 'final_validation_results.json'),
                                                   self.config.num_processes)

        self.logger.info("Validation complete")
        self.logger.info("Mean Validation Dice: %f" % (results_of_val['means_regular']["Dice"]))

        # reset deep supervision in network
        self.network.decoder.deep_supervision = True

    def run_training(self):
        self.on_training_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self._on_training_epoch_start()
            training_outputs = []
            for batch_id in range(self.config.training_iters_per_epoch):
                training_outputs.append(self.train_step(next(self.dataloader_train)))
            self._on_training_epoch_end(training_outputs)

            with torch.no_grad():
                self._on_validation_epoch_start()
                validation_outputs = []
                plot_on_iter = np.random.choice(self.config.val_iters_per_epoch, size = 2)
                for batch_id in range(self.config.val_iters_per_epoch):
                    validation_outputs.append(self.validation_step(next(self.dataloader_val), plot_batch=(batch_id in plot_on_iter)))
                self._on_validation_epoch_end(validation_outputs)

            self.finish_epoch()

        self.finish_training()


if __name__ == "__main__":

    pass