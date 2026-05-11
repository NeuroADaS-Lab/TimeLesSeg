import argparse

from timelesseg.utils import join, maybe_mkdir, basename, remove_nii_extension
from timelesseg.inference import inference_from_list_of_case_dicts, export_prediction_from_probabilities
from timelesseg.trained_models import CHECKPOINTS as _CHK


CHECKPOINTS = (
    _CHK.get('best'),
    _CHK.get('final')
)

def process_multimodality(*images: str,
                          weights,
                          ofolder: str,
                          baseline_mask: str | None,
                          save_probabilities: bool,
                          **inference_kwargs):
    nmod = len(images)

    baseline_masks = [baseline_mask] * nmod

    # prepare inputs for inference
    ofiles = [join(ofolder, basename(remove_nii_extension(f)) + '_timelesseg') for f in images]
    results = inference_from_list_of_case_dicts(
        [{'images': (images[i], baseline_masks[i])} for i in range(nmod)],
        output_files=None,
        save_probabilities=False,
        use_masked_norm=False,
        **inference_kwargs,
    )

    first_modality, *rest_modalities = results
    probs, case_properties = first_modality
    export_prediction_from_probabilities(probs, case_properties, save_probabilities, ofiles[0])
    if nmod == 1:
        return ofiles[0]

    probs *= weights[0]
    for i in range(1, nmod):
        probs_i, case_properties_i = rest_modalities[i-1]
        export_prediction_from_probabilities(probs_i, case_properties_i, save_probabilities, ofiles[i])
        probs += (probs_i * weights[i])

    ofile_ensemble = join(ofolder, basename(remove_nii_extension(images[0])) + '_timelesseg_ensemble')
    export_prediction_from_probabilities(probs,
                                         case_properties,
                                         save_probabilities,
                                         ofile_ensemble)
    return ofile_ensemble


def entrypoint(args = None):
    parser = argparse.ArgumentParser(prog='TimeLesSeg')
    parser.add_argument('-i', '--images', nargs='+', metavar='IMAGE', required=True,
                        help='Input images to segment. Can be one or many.')
    parser.add_argument('-o', '--ofolder', metavar='OUTPUT FOLDER', required=True,
                        help='Folder where segmentations will be left. Each input filename "image.nii.gz" will be segmented and saved '
                             'in [OUTPUT FOLDER] as "[OUTPUT_FOLDER]/image_timelesseg.nii.gz"')
    parser.add_argument('-m', '--baseline-mask', metavar='MASK',
                        help='[Optional] Lesion mask from a previous timepoint to serve as a prior to segment the input scans. '
                             'It is assumed that both timepoints have been registered and thus MASK has the same dimensions as IMAGE.')
    parser.add_argument('-w', '--weights', nargs='+', type=float,
                        help='[Optional] Per-modality weights that will be used to derive the final ensembled segmentation. '
                             'By default we use 1/N where N is the number of input scans.')
    parser.add_argument('--save-probabilities', action='store_true',
                        help='[Optional] Whether the user wants the probabilities for each segmented modality and the final '
                             'ensemble saved to disk (as .npz files). Warning: it consumes hella space.')
    parser.add_argument('-npp', '--num-processes-preprocessing', type=int, default=3,
                        help='[Optional] Number of processes that will be used to parallelize preprocessing. Default: 3.')
    parser.add_argument('-npe', '--num-processes-export', type=int, default=3,
                        help='[Optional] Number of processes that will be used to parallelize the saving of outputs to disk. Default: 3.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='[Optional] Whether you want to be yelled at during the process. Default: False (we have manners).')
    parser.add_argument('--device', default='cpu',
                        help='[Optional] Which device will be used to perform inference. Will be passed on to `torch.device`.')

    args = parser.parse_args(args)
    return args


def main(args = None):
    args = entrypoint(args)

    nimages = len(args.images)
    if args.weights is None:
        args.weights = [1/nimages] * nimages
    else:
        if nimages != len(args.weights):
            msg = (
                'If provided, the number of weights (-w/--weights) should have the same length as IMAGES. '
                'Got %i images and %i weights.' % (nimages, len(args.weights))
            )
            raise ValueError(msg)

    inference_kwargs = {
        'checkpoint_paths': CHECKPOINTS,
        'verbose': args.verbose,
        'num_processes_pp': args.num_processes_preprocessing,
        'num_processes_export': args.num_processes_export,
        'device': args.device
    }

    maybe_mkdir(args.ofolder)

    ofile_ensemble = process_multimodality(*args.images,
                                            weights=args.weights,
                                            ofolder=args.ofolder,
                                            baseline_mask=args.baseline_mask,
                                            save_probabilities=args.save_probabilities,
                                            **inference_kwargs)
    return ofile_ensemble

if __name__ == "__main__":
    main()