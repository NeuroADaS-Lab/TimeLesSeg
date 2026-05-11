from timelesseg.trainer.trainer import Trainer
from timelesseg.config import get_configs
from timelesseg.utils import timestampify, get_default_device


def train(training_config, arch_kwargs, device: str, output_folder: str, checkpoint: str = None):
    trainer = Trainer(training_config, arch_kwargs, device, output_folder)
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)
    trainer.run_training()
    return trainer


def train_entrypoint():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', '--network_architecture', type=str,
                        choices=['unet', 'plainconvunet', 'mednext', 'resunet'],
                        dest='net',
                        default='unet')
    parser.add_argument('-tmem', '--target_memory', type=int,
                        default=64)
    parser.add_argument('-lr', '--learning_rate', type=float,
                        dest='initial_learning_rate')
    parser.add_argument('-wd', '--weight_decay', type=float)
    parser.add_argument('-e', '--expansion_ratio', type=int,
                        dest='expansion_ratio_per_stage')
    parser.add_argument('-pofg', '--p_oversample_foreground', type=float,
                        dest='oversample_fg_probability')
    parser.add_argument('-titer', '--train_iters_per_epoch', type=int,
                        dest='training_iters_per_epoch')
    parser.add_argument('-viter', '--val_iters_per_epoch', type=int)
    parser.add_argument('-nep', '--num_epochs', type=int)
    parser.add_argument('-pcross', '--p_cross_sectional', type=float)
    parser.add_argument('--save_every', type=int,
                        help='Number of epochs between training checkpoint saves.')
    parser.add_argument('-np', '--num_processes', type=int)

    parser.add_argument('--checkpoint', '-c', type=str)


    # this have to be removed from the return dict. However, we could
    # move them to TrainingConfig as well.
    parser.add_argument('--device', type=str, default=get_default_device())
    parser.add_argument('-o', '--output_folder', default=timestampify('timelesseg_trained_models'))

    args = parser.parse_args()

    # remove defaults. We have our defaults stored in timelesseg.configs.{training_config/architecture_config}
    return {k: v for k, v in vars(args).items() if v is not None}


def sanity_check_config(config_from_parser: dict, *configs) -> bool:
    for attr in config_from_parser:
        for c in configs:
            if attr in c.__dict__ and not getattr(c, attr) == config_from_parser[attr]:
                return False
    return True


def main():
    config_overrides = train_entrypoint()
    device, output_folder = config_overrides.pop('device'), config_overrides.pop('output_folder')
    arch_kwargs, training_config, preprocessing_config = get_configs(**config_overrides)
    ok = sanity_check_config(config_overrides, arch_kwargs, training_config, preprocessing_config)
    if not ok:
        raise RuntimeError

    trainer = train(training_config, arch_kwargs, device, output_folder,
                    config_overrides.get('checkpoint'))
    trainer.run_final_validation(preprocessing_config)

if __name__ == "__main__":
    main()