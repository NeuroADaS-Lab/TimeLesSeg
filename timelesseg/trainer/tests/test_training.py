import torch
import numpy as np
import SimpleITK as sitk

from ..trainer import Trainer
from timelesseg.config import get_configs
from timelesseg.dataloading import Dataset


class _TestTrainer(Trainer):

    def run_training(self):
        self.on_training_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            with torch.no_grad():
                self._on_validation_epoch_start()
                validation_outputs = []
                plot_on_iter = np.random.choice(self.config.val_iters_per_epoch, size = 2)
                for batch_id in range(self.config.val_iters_per_epoch):
                    validation_outputs.append(self.validation_step(next(self.dataloader_val), plot_batch=(batch_id in plot_on_iter)))
                self._on_validation_epoch_end(validation_outputs)

class TestTrainer(Trainer):
    def _get_datasets(self):
        tr_dataset = Dataset(self.config.training_data_path)
        val_dataset = Dataset(self.config.validation_data_path)

        if False:
            # this extracts npys from npz files
            self.logger.info('Unpacking training dataset... This may take a while')
            tr_dataset.unpack_dataset(self.config.num_processes, remove_npz=True)

        if False:
            self.logger.info('Unpacking validation dataset...')
            val_dataset.unpack_dataset(self.config.num_processes, remove_npz=False)

        return tr_dataset, val_dataset

    def train_step(self, batch):
        data: torch.Tensor = batch['data']
        target: torch.Tensor | list[torch.Tensor] = batch['target']
        print(batch['keys'])
        if isinstance(target, list):
            target = target[0]
        return data.detach().cpu().numpy(), target.detach().cpu().numpy()

    def run_training(self):
        self.on_training_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self._on_training_epoch_start()
            for batch_id in range(self.config.training_iters_per_epoch):
                bs = self.config.batch_size
                data, target = self.train_step(next(self.dataloader_train))
                assert data.shape[0] == bs
                for b in range(bs):
                    im, baseline = data[b, 0], data[b, 1]
                    print(im.shape)
                    print(baseline.shape)
                    seg = target[b][0]
                    print(seg.shape)
                    assert im.shape == baseline.shape == seg.shape
                    sitk.WriteImage(sitk.GetImageFromArray(im), self.output_folder + '/test_im_batch_%i.nii.gz' % (batch_id + b))
                    sitk.WriteImage(sitk.GetImageFromArray(baseline), self.output_folder + '/test_baseline_batch_%i.nii.gz' % (batch_id + b))
                    sitk.WriteImage(sitk.GetImageFromArray(seg), self.output_folder + '/test_seg_batch_%i.nii.gz' % (batch_id + b))

if __name__ == "__main__":
    arch_kwargs, training_config, preprocessing_config = get_configs('resunet', 36, num_epochs=2)
    tr = TestTrainer(training_config,
                     arch_kwargs,
                     'cpu',
                     'test-training')
    tr.run_training()