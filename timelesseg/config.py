from timelesseg.configs import get_default_training_config, ArchKwargs
from timelesseg.configs.preprocessing import get_preprocessing_config_from_dataset_fingerprint
from timelesseg.experiment_planning import plan_experiment

from .data_stuff import (
    DATASET_FINGERPRINT_PATH,
    TRAINING_PATH,
    VALIDATION_PATH
)


def get_preprocessing_config(dataset_fingerprint_path: str = DATASET_FINGERPRINT_PATH):
    return get_preprocessing_config_from_dataset_fingerprint(dataset_fingerprint_path)

def get_configs(net: str, target_memory: int, dataset_fingerprint_path: str = DATASET_FINGERPRINT_PATH, **config_overrides):
    preprocessing_config = get_preprocessing_config(dataset_fingerprint_path)
    patch_size, arch_kwargs, batch_size, network_class = plan_experiment(preprocessing_config, net, target_memory)
    arch_kwargs = ArchKwargs(**arch_kwargs)
    training_config = get_default_training_config(patch_size=patch_size,
                                                  network_class=network_class,
                                                  batch_size=batch_size,
                                                  training_data_path=TRAINING_PATH,
                                                  validation_data_path=VALIDATION_PATH,
                                                  **config_overrides)
    return arch_kwargs, training_config, preprocessing_config