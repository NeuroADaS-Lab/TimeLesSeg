import numpy as np
import torch

from timelesseg.utils import join, isfile


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    else:
        pass


def get_debug_information_from_trainer(o: object):
    # saving some debug information
    dct = {}
    for k in o.__dir__():

        if not k.startswith("__"):
            this_obj = getattr(o, k)
            if not callable(this_obj) or k == "loss":
                dct[k] = str(this_obj)
                weights = getattr(this_obj, 'weight_factors', None)
                if weights is not None:
                    dct[k + '.weights'] = this_obj.weight_factors
            elif k in ['network', ]:
                dct[k] = str(this_obj.__class__.__name__)
            else:
                # print(k)
                pass
            if k in ['dataloader_train', 'dataloader_val']:
                if hasattr(this_obj, 'generator'):
                    dct[k + '.generator'] = str(this_obj.generator)
                if hasattr(this_obj, 'num_processes'):
                    dct[k + '.num_processes'] = str(this_obj.num_processes)
                if hasattr(this_obj, 'transform'):
                    dct[k + '.transform'] = str(this_obj.transform)

    import subprocess
    hostname = subprocess.getoutput(['hostname'])
    dct['hostname'] = hostname
    torch_version = torch.__version__

    if o.device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        dct['gpu_name'] = gpu_name
        cudnn_version = torch.backends.cudnn.version()
    else:
        cudnn_version = 'None'

    dct['device'] = str(o.device)
    dct['torch_version'] = torch_version
    dct['cudnn_version'] = cudnn_version

    return dct


class dummy_context:
    """For compat with autocast"""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def collate_outputs(outputs: list[dict]) -> dict:
    """
    used to collate default train_step and validation_step outputs. If you want something different then you gotta
    extend this

    we expect outputs to be a list of dictionaries where each of the dict has the same set of keys
    """
    collated = {}
    for k in outputs[0].keys():
        if np.isscalar(outputs[0][k]):
            collated[k] = [o[k] for o in outputs]
        elif isinstance(outputs[0][k], np.ndarray):
            collated[k] = np.vstack([o[k][None] for o in outputs])
        elif isinstance(outputs[0][k], list):
            collated[k] = [item for o in outputs for item in o[k]]
        else:
            raise ValueError(f'Cannot collate input of type {type(outputs[0][k])}. '
                             f'Modify collate_outputs to add this functionality')
    return collated


def has_been_unpacked(dataset_folder, data_identifiers) -> bool:
    super_iterable_bool = map(
        lambda ii: isfile(join(dataset_folder, 'case_%s.npy' % ii)) and isfile(join(dataset_folder, 'case_%s_seg.npy' % ii)),
        data_identifiers
    )
    return all(super_iterable_bool)

def save_necessary_info_for_prediction():
    pass

if __name__ == "__main__":
    from timelesseg.dataloading import Dataset
    from timelesseg.config import TRAINING_PATH
    dataset = Dataset(TRAINING_PATH)
    assert has_been_unpacked(dataset.folder, dataset.identifiers)