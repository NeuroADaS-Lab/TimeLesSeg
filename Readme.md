# TimeLesSeg
## Unified Contrast-Agnostic Cross-Sectional and Longitudinal MS Lesion Segmentation via a Stochastic Generative Model

This repository contains the source code from our publication currently under review, available at [arXiv](https://doi.org/10.48550/arXiv.2605.07955
). In short, our approach unifies cross-sectional (single-timepoint) and longitudinal (multi-timepoint) multiple sclerosis lesion segmentation within a single convolutional neural net. Furthermore, our approach was trained in a fully-randomized & fully-synthetic contrast-agnostic manner [(Billot et al., 2023)](https://doi.org/10.1016/j.media.2023.102789), making it capable of segmenting any MR contrast and resolution.

![TimeLesSeg in action](assets/in-action.png)

Our approach is enabled by three key contributions:
- Modelling longitudinal inputs through lesion masks instead of prior scans.
- Devising a generative longitudinal module that sinthesizes endless prior timepoints given a lesion mask.
- Integrating our longitudinal synthetic module with a previously described GMM-based scan generative approach, for a fully synthetic pipeline that synthesizes a longitudinal MS dataset from a set of less than ten segmentations.

![Generative pipeline](assets/gen-pipeline.png)

### Citation

Our work has been submitted and is currently under review. In the meantime, if you find our work useful please cite our arXiv pre-print:

```
@misc{casellesballester2026timelessegunifiedcontrastagnosticcrosssectional,
      title={TimeLesSeg: Unified Contrast-Agnostic Cross-Sectional and Longitudinal MS Lesion Segmentation via a Stochastic Generative Model}, 
      author={Vicent Caselles-Ballester and Eloy Martínez-Heras and Giuseppe Pontillo and Zoe Mendelsohn and Elena M. Marrón and Juan Luis García Fernández and Laia Subirats and Jon Stutters and Jeremy Chataway and Frederik Barkhof and Sara Llufriu and Ferran Prados},
      year={2026},
      eprint={2605.07955},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.07955}, 
}
```

### Installation
Installation should be straightforward. First, clone the present repo, `cd` into it, and run (with python >= 3.12 installed):
```bash
pip install -e .
```
The `-e` is optional, but it will let you track any changes you might make to the code.

### Usage

TimeLesSeg can process an arbitrary number of scans, provided that they belong to the same subject and all have been registered (are all in the same space). Using an optional baseline mask, one can provide the model with a prior on disease state, equating to longitudinal processing.

```bash
python3 entrypoint.py \
    -i [SCAN 1] [SCAN 2] ... [SCAN M] \
    -o [OUTPUT FOLDER] \
    -m [BASELINE MASK] (OPTIONAL) \
    -w [WEIGHT 1] [WEIGHT 2] ... [WEIGHT M] (OPTIONAL) \
    -device [DEVICE] (OPTIONAL)
```

Using the `-w` flag, you can control how the probabilities derived from each modality are combined. For example, given that FLAIR represents the clinical gold standard to identify MS lesions, one might desire to upweight it w.r.t. other modalities such as T1w.


### Acknowledgements

The neural network configuration, model architecture, and training code are copied from nnUNet, a state of the art self-configuring biomedical image segmentation framework [(Isensee et al., 2021)](https://doi.org/10.1038/s41592-020-01008-z
). For data augmentation, I have also used (and extended a little bit) their batchgenerators framework (see `pyproject.toml`).

The [singleton pattern](timelesseg/singleton/) was inspired by [Francesco Galati](https://github.com/i-vesseg/MultiVesSeg/blob/b7472b756a73d62f3163725ae7d5a5968b81766a/phase2/options/singleton.py).