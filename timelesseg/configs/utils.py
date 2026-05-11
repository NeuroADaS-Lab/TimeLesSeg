import torch

def get_configs_from_checkpoint(checkpoint: str | dict) -> tuple:
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint)
    trainer_init_args = checkpoint['init_args']
    return trainer_init_args['arch_kwargs'], trainer_init_args['training_config']