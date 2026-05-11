import torch
import numpy as np

from timelesseg.data_stuff import LABELS


def apply_nonlinearity_to_logits(logits_array: np.ndarray | torch.Tensor) -> torch.Tensor:

    if isinstance(logits_array, np.ndarray):
        logits_array = torch.from_numpy(logits_array)

    with torch.no_grad():
        probabilities = torch.softmax(logits_array.float(), dim=0)

    return probabilities


@torch.inference_mode()
def convert_probabilities_to_segmentation(probabilities_array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    assumes that nonlinearity was already applied!
    probabilities_array has to have shape (c, x, y(, z)) where c is the number of classes
    """
    if probabilities_array.shape[0] != len(LABELS):
        msg = (
            'Unexpected number of channels in predicted_probabilities. '
            'Expected %i, got %i. Remember that predicted_probabilities should have shape (c, x, y(, z)).' % \
                (probabilities_array.shape[0], len(LABELS))
        )
        raise ValueError(msg)

    # assert probabilities_array.shape[0] == 2, 
    #     probabilities_array.min() >= - 1e-6 and
    #     probabilities_array.max() < (1. + 1e-6)
    # )
    return probabilities_array.argmax(0)


@torch.inference_mode()
def convert_logits_to_segmentation(
    predicted_logits: np.ndarray | torch.Tensor,
    return_probabilities: bool
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
    """
    :return: If the input (predicted_logits) is of type A, the output will be all of type A
    """

    probabilities = apply_nonlinearity_to_logits(predicted_logits)
    if isinstance(predicted_logits, np.ndarray):
        probabilities = probabilities.cpu().numpy()

    seg = convert_probabilities_to_segmentation(probabilities)

    if not return_probabilities:
        probabilities = None

    return seg, probabilities
