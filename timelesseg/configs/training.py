from dataclasses import dataclass
import torch
from torch import nn

from timelesseg.typing import ArrayLike
from timelesseg.trainer.scheduler import PolyLRScheduler

from .abstract import Config


@dataclass
class TrainingConfig(Config):
    patch_size: ArrayLike
    network_class: nn.Module
    optim_class: nn.Module
    scheduler_class: nn.Module
    batch_size: int
    initial_learning_rate: float
    weight_decay: float
    optim_kwargs: dict
    oversample_fg_probability: float
    training_iters_per_epoch: int
    val_iters_per_epoch: int
    num_epochs: int
    deep_supervision: bool
    n_deep_supervision_stages: int
    training_data_path: str
    validation_data_path: str
    num_processes: int
    p_cross_sectional: float
    save_every: int


class JoinedConfigs(Config):
    def __init__(self, *configs: Config):
        self._join_configs_init(*configs)
        self.__configs = {c.__class__.__name__: c for c in configs}

    def _join_configs_init(self, *configs: Config):
        seen = set()
        for config in configs:
            self.update(config)
            if any(k in seen for k in config.keys()) and not all(config[k] == self[k] for k in config.keys() if k in seen):
                raise ValueError
            seen |= set(config.keys())

    def join_configs(self, config):
        seen = set(self.keys())
        if any(k in seen for k in config.keys()) and not all(config[k] == self[k] for k in config.keys() if k in seen):
            raise ValueError
        self.update(config)

    def __add__(self, other: Config):
        assert issubclass(other.__class__, Config)
        self.join_configs(other)
        return self

    def get_config(self, name: str):
        return self.__configs[name]


def get_default_training_config(**overrides):
    default_training_kwargs = {
        'patch_size': None,
        'batch_size': None,
        'network_class': None,
        'training_data_path': None,
        'validation_data_path': None,
        'optim_class': torch.optim.SGD,
        'scheduler_class': PolyLRScheduler,
        'initial_learning_rate': 1e-2,
        'weight_decay': 3e-5,
        'optim_kwargs': {'momentum': 0.99, 'nesterov': True},
        'oversample_fg_probability': 0.33,
        'training_iters_per_epoch': 250,
        'val_iters_per_epoch': 50,
        'num_epochs': 1000,
        'deep_supervision': True,
        'n_deep_supervision_stages': 2,
        'num_processes': 12,
        'p_cross_sectional': 0.3,
        'save_every': 50
    }
    training_kwargs = default_training_kwargs | {k: overrides[k] for k in overrides if k in default_training_kwargs}
    return TrainingConfig(**training_kwargs)


if __name__ == "__main__":

    training_config = get_default_training_config()
    print(training_config)