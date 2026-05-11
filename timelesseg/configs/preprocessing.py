from timelesseg.typing import ArrayLike, ToIterableInt
from timelesseg.utils import load_json
from timelesseg.singleton import Singleton
from timelesseg.experiment_planning.utils import determine_target_spacing, determine_transpose


class PrepConfig(metaclass=Singleton):

    def __init__(
        self,
        dataset_fingerprint: dict,
        target_spacing: ArrayLike,
        transpose_forward: ToIterableInt,
        transpose_backward: ToIterableInt
    ):
        self._dataset_fingerprint = dataset_fingerprint
        self._target_spacing = target_spacing
        self._transpose_forward = transpose_forward
        self._transpose_backward = transpose_backward
        self._spacing_has_been_transposed = False

    @classmethod
    def from_data_fingerprint(cls, dataset_fingerprint: str | dict):
        if isinstance(dataset_fingerprint, str):
            dataset_fingerprint = load_json(dataset_fingerprint)
        target_spacing = determine_target_spacing(dataset_fingerprint['spacings'],
                                                  dataset_fingerprint['shapes_after_cropping'])
        transpose_fwd, transpose_bwd = determine_transpose(target_spacing, suppress_transpose=False)
        return cls(dataset_fingerprint, target_spacing, transpose_fwd, transpose_bwd)

    @property
    def dataset_fingerprint(self):
        return self._dataset_fingerprint

    @property
    def target_spacing(self):
        if self._spacing_has_been_transposed:
            return self._target_spacing
        # only do it one time
        self._target_spacing[self.transpose_forward]
        self._spacing_has_been_transposed = True
        return self._target_spacing

    @property
    def transpose_forward(self):
        return self._transpose_forward

    @property
    def transpose_backward(self):
        return self._transpose_backward

    def __str__(self):
        return {k: getattr(self, k) for k in self.__dir__() if k in ['transpose_forward', 'transpose_backward', 'target_spacing']}.__str__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        # let's make use of this beautiful method we've written
        return str(self) == str(other)

def get_preprocessing_config_from_dataset_fingerprint(dataset_fingerprint: str | dict):
    """This is to avoid circular imports. Don't import PrepConfig directly!"""
    return PrepConfig.from_data_fingerprint(dataset_fingerprint)


if __name__ == "__main__":
    config = PrepConfig.from_data_fingerprint('training_data/raw/dataset_fingerprint.json')
    assert not config._spacing_has_been_transposed
    target_spacing = config.target_spacing
    assert config._spacing_has_been_transposed

    config2 = PrepConfig.from_data_fingerprint('training_data/raw/dataset_fingerprint_old.json')
    assert config == config2