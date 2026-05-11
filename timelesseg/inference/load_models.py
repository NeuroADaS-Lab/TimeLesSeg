import pickle
import torch


# I am so fucking stupid. I saved my models when my project folder was timelessegv2, which I've now changed
# to timelesseg, so now we have to catch this error during calling torch.load
class MyPickleModule:
    class RenameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('timelessegv2'):
                module = module.replace('timelessegv2', 'timelesseg')

            return super().find_class(module, name)

    Unpickler = RenameUnpickler

def load_model(f, **kwargs):
    assert 'pickle_module' not in kwargs
    return torch.load(f, pickle_module=MyPickleModule, **kwargs)

