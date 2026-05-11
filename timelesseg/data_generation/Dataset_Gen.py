import logging
import time
import argparse
from glob import glob
from typing import List, Generator, Optional, Callable
from functools import partial
import numpy as np

from timelesseg.typing import ArrayLike
from timelesseg.utils import (
    join,
    basename,
    dirname,
    maybe_mkdir,
    TmpDir,
    check_input_lists,
    setup_loggers
)
from .utils import map_seg
from .image import (
    load_nifti,
    save_nifti
)
from timelesseg.data_generation import fake_lesion_mask as flm
from timelesseg.data_generation.scan_generator import ScanGenerator
from timelesseg.data_generation import labeling_convention as lab
from .ensure_differences import percentile_based_criteria_with_std_no_sort


logger = logging.getLogger(__name__)

class DatasetGenerator:

    LOGGERS_TO_ACTIVATE = (logger.name, flm.__name__, ScanGenerator.__module__)

    def __init__(
        self,
        output_folder: str,
        lesion_masks: List[str],
        parcellations: List[str],
        pseudo_mask_iters: int,
        synthseg_iters: int,
        fake_lesion_mask_iters: int,
        log_file: str = None,
        verbosity: str = None,
        start_from_index: int = 1
    ):

        self.output_folder = output_folder

        if not check_input_lists(lesion_masks, parcellations, f = lambda x: basename(dirname(x))):
            raise ValueError(f'Input lists do not match. Got:\n' + "\n".join(lesion_masks + parcellations))

        self.lesion_masks = lesion_masks
        self.parcellations = parcellations
        self._num_initial_gts = len(self.lesion_masks)

        self.pseudo_mask_iters = pseudo_mask_iters
        self.synthseg_iters = synthseg_iters
        self.fake_lesion_mask_iters = fake_lesion_mask_iters

        # used for generating previous timepoints given mask at t
        self.plausible_fake_lesion_mask: Callable[[np.ndarray, ArrayLike], np.ndarray] = partial(
            flm.fake_lesion_mask, params=flm.PLAUSIBLE_PARAMS
        )

        # used for augmenting lesion masks
        self.crazy_fake_lesion_mask: Callable[[np.ndarray, ArrayLike], np.ndarray] = partial(
            flm.fake_lesion_mask, params=flm.CRAZY_PARAMS
        )

        self._t_stat_thr = 0.01 # UNUSED

        self._WM_code = lab._cerebral_WM
        self._MS_les_code = lab.MS_lesion
        self._segmentation_targets = lab.SEGMENTATION_TARGETS
        self._skull_stripping_classes = lab.SKULL_STRIPPING_CLASSES

        # places were lesions cannot be (background included!)
        # because lesions cannot suddenly appear floating outside brain dummy
        self._MS_les_nono = [0, lab._ventricular_CSF, lab._outer_CSF, lab._non_brain_high, lab._non_brain_mid, lab._non_brain_low]

        self._current_im = start_from_index
        self._num_cases_to_produce = self._num_initial_gts * self.pseudo_mask_iters * self.synthseg_iters * self.fake_lesion_mask_iters
        self._f_rjust = lambda x: str(x).rjust(len(str(self._num_cases_to_produce)), "0")

        self._initialize_loggers_and_output_folder(log_file, verbosity)

    def _initialize_loggers_and_output_folder(self, log_file: Optional[str], verbosity: Optional[str]) -> None:

        maybe_mkdir(self.output_folder)
        if log_file is None:
            log_file = f'dataset_gen_log_{time.strftime("%d%m%Y_%H%M%S")}.txt'
            log_file = join(self.output_folder, log_file)

        if verbosity is None:
            verbosity = 'DEBUG'
            _console_verbosity = 'INFO'
        else: _console_verbosity = verbosity

        setup_loggers(*self.LOGGERS_TO_ACTIVATE,
                      verbosity=verbosity,
                      log_file=log_file,
                      console_verbosity=_console_verbosity)

    def _get_index_for_current_im(self) -> str:
        """
        Processes current im as str and adds one.
        """
        index = self._f_rjust(self._current_im)
        self._current_im += 1
        return index

    def _initialize_brain_generator(self, labels_dir: str) -> ScanGenerator:
        synthseg_kwargs = {
            # max_res_aniso controls the upper bound of the anisotropic resolution dimension.
            'max_res_aniso': 5.,

            # dim_rand_res controls which dimension will be anisotropic.
            # Remember, SynthSeg brings everything to RAS coordinates, therefore
            # you have to pass index in accordance to it. (I pass 0 since I want sagittal)
            # 'dim_rand_res': 0,

            'normalise': None,

            # turn off gamma augmentation -- will do it during dataloading
            'gamma_std': 0.
        }
        # construct the set of values in parcellation. We add background and lesions code.
        classes_in_parcellation_w_lesions = set(lab.MAPPING_FROM_GIF_TO_INTERNAL.values()) | {0, self._MS_les_code}
        difference_function = partial(
            percentile_based_criteria_with_std_no_sort,
            wm_class = self._WM_code,
            lesions_class = self._MS_les_code,
            percentile = 50,
            class_vals = classes_in_parcellation_w_lesions
        )
        sg =  ScanGenerator(labels_dir,
                            brain_generator_kwargs=synthseg_kwargs,
                            wm_val=self._WM_code,
                            les_val=self._MS_les_code,
                            difference_function=difference_function)
        return sg

    def augment_initial_lesion_mask(self, lesion_mask: np.ndarray, spacing: ArrayLike):
        for i in range(self.pseudo_mask_iters):
            logger.debug('Running iteration %i of %i of fake_lesion_mask with crazy parameters.', i + 1, self.pseudo_mask_iters)
            augmented_mask = self.crazy_fake_lesion_mask(lesion_mask, spacing)
            yield augmented_mask

    def _process_parcellation(self, parcellation_path: str) -> np.ndarray:
        """Loads original parcellation as outputted by GIF and maps it to internal simpler classes."""
        parcellation = load_nifti(parcellation_path)
        parcellation_mapped = map_seg(parcellation.data, lab.MAPPING_FROM_GIF_TO_INTERNAL)
        return parcellation_mapped

    def _join_lesion_mask_and_parcellation(
        self,
        lesion_mask: np.ndarray,
        parcellation: np.ndarray,
        mask_where_lesions_cannot_be: np.ndarray = None
    ) -> np.ndarray:

        # first check that no lesion mask is where it shouldn't
        if mask_where_lesions_cannot_be is None:
            mask_where_lesions_cannot_be = np.isin(parcellation, self._MS_les_nono)

        # intersect lesion_mask with the inverse of where lesions cannot be
        # inverse of lesions CANNOT be --> where lesions CAN be
        lesion_mask &= ~mask_where_lesions_cannot_be
        lesion_mask = map_seg(lesion_mask, {1: self._MS_les_code}).astype(parcellation.dtype)

        assert self._MS_les_code not in set(np.unique(parcellation.ravel()))

        parcellation = np.where(lesion_mask != 0, 0, parcellation)

        return lesion_mask + parcellation


    def _run_fake_lesion_mask(
        self,
        lesion_mask: np.ndarray,
        spacing: ArrayLike,
        mask_where_lesions_cannot_be: np.ndarray
    ):
        for i in range(self.fake_lesion_mask_iters):
            logger.debug('Running iteration %i of %i of fake_lesion_mask with plausible parameters.', i + 1, self.fake_lesion_mask_iters)
            augmented_mask = self.plausible_fake_lesion_mask(lesion_mask, spacing)

            augmented_mask &= ~mask_where_lesions_cannot_be

            yield augmented_mask

    def _run_synthseg(self, seg_path: str):
        scan_generator = self._initialize_brain_generator(seg_path)
        for i in range(self.synthseg_iters):
            logger.debug('Generating image %i of %i with SynthSeg.', i + 1, self.synthseg_iters)
            # each call to ScanGenerator returns:
            # scan, lesion_mask, affine, header
            yield scan_generator()

    def save_outputs(self, scan: np.ndarray, lesion_mask: np.ndarray, baseline_lesion_masks: Generator,
                     affine: np.ndarray, header):

        saver = partial(save_nifti, affine=affine, header=header)

        case_index = self._get_index_for_current_im()

        saver(scan, save_path=join(self.output_folder, f'If_{case_index}.nii.gz'))
        saver(lesion_mask, save_path=join(self.output_folder, f'Mf_{case_index}.nii.gz'))

        for b, Mb in enumerate(baseline_lesion_masks):
            Mb = Mb.astype(np.uint8, copy=False)
            saver(Mb, save_path=join(self.output_folder, f'Mb{b+1}_{case_index}.nii.gz'))

    def run_pipeline_one(self, lesion_mask: str, parcellation: str):
        initial_mask = load_nifti(lesion_mask)
        ndim = initial_mask.ndim

        augmented_masks = self.augment_initial_lesion_mask(initial_mask.data, initial_mask.spacing)

        p = self._process_parcellation(parcellation)
        # compute this here to avoid computing it many times for no reason
        mask_where_lesions_cannot_be = np.isin(p, self._MS_les_nono)

        tmpdirname = '.DatasetGen__'
        for augmented_mask in augmented_masks:
            augmented_mask = self._join_lesion_mask_and_parcellation(augmented_mask, p, mask_where_lesions_cannot_be)

            with TmpDir(root=tmpdirname) as tmpdir:
                myseg_for_synthseg = join(tmpdir, 'tmp_for_synthseg.nii.gz')
                save_nifti(augmented_mask, save_path=myseg_for_synthseg, affine=initial_mask.affine, header=initial_mask.header)
                generated_cases = self._run_synthseg(myseg_for_synthseg)
                for (im, seg, affine, header) in generated_cases:
                    spacing = header['pixdim'][1:ndim+1]
                    lesmask = seg == self._MS_les_code
                    # recalculate it here because synthseg applies nonlinear transformations!!!!
                    new_mask_where_lesions_cannot_be = np.isin(seg, self._MS_les_nono)
                    baseline_masks = self._run_fake_lesion_mask(lesmask, spacing, new_mask_where_lesions_cannot_be)
                    self.save_outputs(im, lesmask, baseline_masks, affine, None)

    def run(self):

        logger.info(f"Hello! I am going to generate {self._num_cases_to_produce} images starting "
                    f"from {self._num_initial_gts} segmentations. Results will be left at {self.output_folder}.")

        logger.debug('Input ground truths (lesion mask, parcellation):\n%s',
                     '\n'.join(['\t'.join([l, p]) for l, p in zip(self.lesion_masks, self.parcellations)]))

        for gt in range(self._num_initial_gts):
            logger.info('Running the pipeline on lesion mask "%s" and parcellation "%s"',
                        self.lesion_masks[gt], self.parcellations[gt])
            self.run_pipeline_one(self.lesion_masks[gt], parcellation=self.parcellations[gt])


def parse_inputs(lesion_masks: list[str], parcellations: list[str]):
    if len(lesion_masks) == 1:
        lesion_masks = glob(lesion_masks[0])
    if len(parcellations) == 1:
        parcellations = glob(parcellations[0])
    return lesion_masks, parcellations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lesion_masks', nargs='+', required=True)
    parser.add_argument('--parcellations', nargs='+', required=True)
    parser.add_argument('--output_folder', '-o', type=str, required=True)
    parser.add_argument('--pseudo_mask_iters', '-pi', type=int, default=15)
    parser.add_argument('--synthseg_iters', '-si', type=int, default=25)
    parser.add_argument('--fake_lesion_mask_iters', '-flmi', type=int, default=5)
    parser.add_argument('--log_file', '-l', type=str, nargs=1, default=None)
    parser.add_argument('--verbosity', '-v', type=str,
                        choices=[i for i in logging._levelToName.values() if i != 'NOTSET'],
                        default='DEBUG')
    parser.add_argument('--start_from_index', type=int, default=1)
    args = parser.parse_args()
    kwargs = vars(args)
    lesion_masks, parcellations = parse_inputs(args.pop('lesion_masks'), args.pop('parcellations'))
    dgen = DatasetGenerator(args.pop('output_folder'), lesion_masks, parcellations, **kwargs)
    dgen.run()

if __name__ == "__main__":
    main()