import torch


from timelesseg.dataloading import Dataset
from timelesseg.config import get_configs
from ..dice import MemoryEfficientSoftDicewWeightsLoss
from ..loss import DC_and_CE_loss

class DummyWeightedDice(MemoryEfficientSoftDicewWeightsLoss):
    def get_wfunc(self, weight_type):
        def func(gt):
            func_results = torch.reciprocal(gt * gt)
            print(func_results)
            print([func_results[:, 0] / func_results[:, i] for i in range(1, 7)])
            return func_results
        return func


class TestTrainer(WeightedDiceTrainer):
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
        prediction = torch.randn((data.shape[0], self.arch_kwargs.num_classes, *data.shape[2:]))
        if isinstance(target, list):
            target = target[0]
        self.loss(prediction, target)
    
    def build_loss(self, deep_supervision, n_stages, n_deep_supervision_stages):

        dice_kwargs = {
            'batch_dice': False,
            'do_bg': False,
            'smooth': 1e-5,
            'weight_type': 'square' # ONE OF ['square', 'uniform', 'simple']
        }
        ce_kwargs = {}
        # TODO: SoftXORCELoss?
        loss = DC_and_CE_loss(dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, dice_class=DummyWeightedDice)

        return loss


if __name__ == "__main__":
    arch_kwargs, training_config, preprocessing_config = get_configs('resunet', 36, num_epochs=2, training_iters_per_epoch=5, val_iters_per_epoch=2, num_processes=2,
                                                                     device='mps',
                                                                     deep_supervision=False)
    tr = TestTrainer(training_config,
                     arch_kwargs,
                     'cpu',
                     'test-training')
    tr.run_training()