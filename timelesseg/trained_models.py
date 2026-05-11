import argparse

from .utils import join


MODELS_FOLDER = 'trained_models/resunet_128_128_96_20_09_25'
CHECKPOINTS = {
    'best': join(MODELS_FOLDER, 'checkpoint_best.pth'),
    'final': join(MODELS_FOLDER, 'checkpoint_final.pth'),
    'early': join(MODELS_FOLDER, 'checkpoint_best_pre_1000_epochs.pth')
}
DEFAULT_CHK_PATH = CHECKPOINTS['best']

class CheckpointAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string):
        mapped_value = [CHECKPOINTS.get(v, v) for v in value] if isinstance(value, list) else CHECKPOINTS.get(value, value)
        setattr(namespace, self.dest, mapped_value)
