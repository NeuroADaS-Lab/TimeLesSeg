from typing import Callable, Tuple
import logging
import numpy as np

from SynthSeg.brain_generator import BrainGenerator

logger = logging.getLogger(__name__)

class ScanGenerator:
    """
    wrapper around BrainGenerator

    :param difference_function: should return True if the two arrays are actually NOT different
    """
    def __init__(
        self,
        seg_path: str,
        brain_generator_kwargs: dict,
        wm_val: int = None,
        les_val: int = None,
        difference_function: Callable[[Tuple[np.ndarray, np.ndarray]], bool] = None,
    ):
        self.brain_generator = BrainGenerator(seg_path, **brain_generator_kwargs)
        self._wm_val = wm_val
        self._les_val = les_val
        self._difference_function = difference_function

    def __call__(self):

        im_i, seg_i = self.brain_generator.generate_brain()
        lesion_mask = seg_i == self._les_val

        if self._difference_function is None:
            return im_i, lesion_mask, self.brain_generator.aff, self.brain_generator.header

        assert self._wm_val is not None and self._les_val is not None

        if lesion_mask.any():

            not_different = self._difference_function(im_i, seg_i)
            # we repeat the process until we have a "valid" image
            counter_repeat = 0
            while not_different:

                im_i, seg_i = self.brain_generator.generate_brain()

                not_different = self._difference_function(im_i, seg_i)

                counter_repeat += 1

            if counter_repeat > 0:
                logger.debug('The initial synthetically generated image did not show sufficient differences '
                             'between lesions and other tissues (WM). It was regenerated %i times.', counter_repeat)

            wm_mean = im_i[seg_i == self._wm_val].mean()
            les_mean = im_i[seg_i == self._les_val].mean()
            logger.debug('Accepted scan with lesions. The means of white matter, MS lesions, respectively are: %f %f',
                         wm_mean, les_mean)

        return im_i, seg_i, self.brain_generator.aff, self.brain_generator.header