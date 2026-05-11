import os
from timelesseg.utils import load_yaml, abspath

YAML_CONFIG = os.environ.get('CONFIG_YAML', 'config.yaml')

config = load_yaml(YAML_CONFIG)

data_config = config['data']
DATASET_FINGERPRINT_PATH = data_config['dataset_fingerprint']
LABELS = list(range(data_config['num_classes']))

training_config = config['training']

try:
    TRAINING_PATH = abspath(training_config['training_data'])
except TypeError:
    TRAINING_PATH = None

try:
    VALIDATION_PATH = abspath(training_config['validation_data'])
except TypeError:
    VALIDATION_PATH = None
