from typing import Set, Dict

####
# INTERNALS
# no left/right assymetry
MS_lesion = 1
_cerebral_WM = 2
_cerebral_GM = 3
_cerebellar_WM = 4
_cerebellar_GM = 5
_thalamus = 6
_caudate = 7
_putamen_and_pallidum = 8
_ventral_DC = 9
_hippocampus = 10
_ventricular_CSF = 11
_brain_stem_and_pons = 12
_outer_CSF = 13
_non_brain_mid = 14
_non_brain_high = 15
_non_brain_low = 16

SEGMENTATION_TARGETS = [
    MS_lesion
]

SKULL_STRIPPING_CLASSES = [
    _non_brain_low,
    _non_brain_mid,
    _non_brain_high
]

# left/right classes (now abandoned)
# internals (inspired by SynthSeg https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/data/labels%20table.txt)
_left_WM = 2
_left_GM = 3
_right_WM = 41
_right_GM = 42
_right_thalamus = 49
_left_thalamus = 10
_right_ventral_DC = 60
_left_ventral_DC = 28

####
# GIF (Geodesic Information Flow)
# the following codes come from the convention at http://niftyweb.cs.ucl.ac.uk/data/BrainParcellationAreas-GIF_v3.xlsx
# and http://niftyweb.cs.ucl.ac.uk/data/labels_v3.xml (XML code: 1=CSF, 2=GM, 3=WM, 4=DGM, 5=Brain stem & pons, >5=non-brain)

## WHITE MATTER
GIF_right_white_matter = {
    81, # Right Temporal White Matter
    82, # Right Insula White Matter
    83, # Right Cingulate White Matter
    84, # Right Frontal White Matter
    85, # Right Occipital White Matter
    86  # Right Parietal White Matter
}
GIF_left_white_matter = {
    89, # Left Temporal White Matter
    90, # Left Insula White Matter
    91, # Left Cingulate White Matter
    92, # Left Frontal White Matter
    93, # Left Occipital White Matter
    94  # Left Parietal White Matter
}

## GRAY MATTER
GIF_right_gray_matter = {
    101, # Right ACgG anterior cingulate gyrus
    103, # Right AIns anterior insula
    105, # Right AOrG anterior orbital gyrus
    107, # Right AnG angular gyrus
    109, # Right Calc calcarine cortex
    113, # Right CO central operculum
    115, # Right Cun cuneus
    117, # Right Ent entorhinal area
    119, # Right FO frontal operculum
    121, # Right FRP frontal pole
    123, # Right FuG fusiform gyrus
    125, # Right GRe gyrus rectus
    129, # Right IOG inferior occipital gyrus
    133, # Right ITG inferior temporal gyrus
    135, # Right LiG lingual gyrus
    137, # Right LOrG lateral orbital gyrus
    139, # Right MCgG middle cingulate gyrus
    141, # Right MFC medial frontal cortex
    143, # Right MFG middle frontal gyrus
    145, # Right MOG middle occipital gyrus
    147, # Right MOrG medial orbital gyrus
    149, # Right MPoG postcentral gyrus medial segment
    151, # Right MPrG precentral gyrus medial segment
    153, # Right MSFG superior frontal gyrus medial segment
    155, # Right MTG middle temporal gyrus
    157, # Right OCP occipital pole
    161, # Right OFuG occipital fusiform gyrus
    163, # Right OpIFG opercular part of the inferior frontal gyrus
    165, # Right OrIFG orbital part of the inferior frontal gyrus
    167, # Right PCgG posterior cingulate gyrus
    169, # Right PCu precuneus
    171, # Right PHG parahippocampal gyrus
    173, # Right PIns posterior insula
    175, # Right PO parietal operculum
    177, # Right PoG postcentral gyrus
    179, # Right POrG posterior orbital gyrus
    181, # Right PP planum polare
    183, # Right PrG precentral gyrus
    185, # Right PT planum temporale
    187, # Right SCA subcallosal area
    191, # Right SFG superior frontal gyrus
    193, # Right SMC supplementary motor cortex
    195, # Right SMG supramarginal gyrus
    197, # Right SOG superior occipital gyrus
    199, # Right SPL superior parietal lobule
    201, # Right STG superior temporal gyrus
    203, # Right TMP temporal pole
    205, # Right TrIFG triangular part of the inferior frontal gyrus
    207  # Right TTG transverse temporal gyrus
}
GIF_left_gray_matter = {
    102, # Left ACgG anterior cingulate gyrus
    104, # Left AIns anterior insula
    106, # Left AOrG anterior orbital gyrus
    108, # Left AnG angular gyrus
    110, # Left Calc calcarine cortex
    114, # Left CO central operculum
    116, # Left Cun cuneus
    118, # Left Ent entorhinal area
    120, # Left FO frontal operculum
    122, # Left FRP frontal pole
    124, # Left FuG fusiform gyrus
    126, # Left GRe gyrus rectus
    130, # Left IOG inferior occipital gyrus
    134, # Left ITG inferior temporal gyrus
    136, # Left LiG lingual gyrus
    138, # Left LOrG lateral orbital gyrus
    140, # Left MCgG middle cingulate gyrus
    142, # Left MFC medial frontal cortex
    144, # Left MFG middle frontal gyrus
    146, # Left MOG middle occipital gyrus
    148, # Left MOrG medial orbital gyrus
    150, # Left MPoG postcentral gyrus medial segment
    152, # Left MPrG precentral gyrus medial segment
    154, # Left MSFG superior frontal gyrus medial segment
    156, # Left MTG middle temporal gyrus
    158, # Left OCP occipital pole
    162, # Left OFuG occipital fusiform gyrus
    164, # Left OpIFG opercular part of the inferior frontal gyrus
    166, # Left OrIFG orbital part of the inferior frontal gyrus
    168, # Left PCgG posterior cingulate gyrus
    170, # Left PCu precuneus
    172, # Left PHG parahippocampal gyrus
    174, # Left PIns posterior insula
    176, # Left PO parietal operculum
    178, # Left PoG postcentral gyrus
    180, # Left POrG posterior orbital gyrus
    182, # Left PP planum polare
    184, # Left PrG precentral gyrus
    186, # Left PT planum temporale
    188, # Left SCA subcallosal area
    192, # Left SFG superior frontal gyrus
    194, # Left SMC supplementary motor cortex
    196, # Left SMG supramarginal gyrus
    198, # Left SOG superior occipital gyrus
    200, # Left SPL superior parietal lobule
    202, # Left STG superior temporal gyrus
    204, # Left TMP temporal pole
    206, # Left TrIFG triangular part of the inferior frontal gyrus
    208  # Left TTG transverse temporal gyrus
}

GIF_pallidum = {
    56, # Right Pallidum
    57  # Left Pallidum
}

GIF_putamen = {
    58, # Right Putamen
    59  # Left Putamen
}

GIF_caudate = {
    37, # Right Caudate
    38  # Left Caudate
}

GIF_right_deep_gray_matter = {
    24, # Right Accumbens Area
    64, # Right Vessel
    66, # Right Ventricular Lining
    77, # Right Basal Forebrain
    96  # Right Claustrum
}

GIF_left_deep_gray_matter = {
    31, # Left Accumbens Area
    65, # Left Vessel
    67, # Left Ventricular Lining
    76, # Left Basal Forebrain
    97  # Left Claustrum
}
GIF_right_cerebellum_gray_matter = {
    39 # Right Cerebellum Exterior
}

GIF_left_cerebellum_gray_matter = {
    40 # Left Cerebellum Exterior
}

GIF_right_cerebellum_white_matter = {
    41 # Right Cerebellum White Matter
}

GIF_left_cerebellum_white_matter = {
    42 # Left Cerebellum White Matter
}

GIF_right_thalamus = {
    60 # Right Thalamus Proper
}
GIF_left_thalamus = {
    61 # Left Thalamus Proper
}

GIF_right_ventral_DC = {
    62
}
GIF_left_ventral_DC = {
    63
}
GIF_right_hippocampus = {
    48
}
GIF_left_hippocampus = {
    49
}
GIF_right_amygdala = {
    32
}
GIF_left_amygdala = {
    33
}

GIF_optic_chiasm = {
    70
}

GIF_Cerebellar_Vermal_Lobules = {
    72, # Cerebellar Vermal Lobules I-V
    73, # Cerebellar Vermal Lobules VI-VII
    74  # Cerebellar Vermal Lobules VIII-X
}

GIF_corpus_callosum = {
    87
}

# ventricles and csf
GIF_left_ventricle = {
    51, # left inferior lateral ventricle
    53  # left lateral ventricle
}
GIF_right_ventricle = {
    52, # right lateral ventricle
    50  # right inferior lateral ventricle
}
GIF_third_ventricle = {
    5, # 3rd ventricle
    47 # 3rd ventricle posterior part
}
GIF_fourth_ventricle = {
    12 # 4th ventricle
}
GIF_fifth_ventricle = {
    16 # 5th ventricle
}
GIF_non_ventricular_CSF = {
    4,  # non-ventricular CSF,
    # I don't know where to put these two
    43, # Right Cerebral Exterior
    44  # Left Cerebral Exterior
}
GIF_ventricles = {
    *GIF_third_ventricle,
    *GIF_fourth_ventricle,
    *GIF_fifth_ventricle,
    *GIF_non_ventricular_CSF,
    *GIF_left_ventricle,
    *GIF_right_ventricle
}

GIF_pons = {
    35
}
GIF_Brain_stem = {
    36
}

GIF_Non_brain_low = {
    1 # non-brain low
}
GIF_Non_brain_mid = {
    2 # non-brain mid
}
GIF_Non_brain_high = {
    3 # non-brain high
}

GIF_all_labels = {
    *GIF_left_ventral_DC,
    *GIF_right_ventral_DC,
    *GIF_left_thalamus,
    *GIF_right_thalamus,
    *GIF_right_hippocampus,
    *GIF_left_hippocampus,
    *GIF_ventricles,
    *GIF_Non_brain_low,
    *GIF_Non_brain_mid,
    *GIF_Non_brain_high,
    *GIF_left_cerebellum_gray_matter,
    *GIF_right_cerebellum_gray_matter,
    *GIF_right_cerebellum_white_matter,
    *GIF_left_cerebellum_white_matter,
    *GIF_right_gray_matter,
    *GIF_left_gray_matter,
    *GIF_right_white_matter,
    *GIF_left_white_matter,
    *GIF_right_deep_gray_matter,
    *GIF_left_deep_gray_matter,
    *GIF_right_amygdala,
    *GIF_left_amygdala,
    *GIF_Brain_stem,
    *GIF_pons,
    *GIF_optic_chiasm,
    *GIF_corpus_callosum,
    *GIF_Cerebellar_Vermal_Lobules,
    *GIF_caudate,
    *GIF_pallidum,
    *GIF_putamen
}


def construct_mapping_from_sets(super_big_set: Set[int]) -> Dict[int, int]:
    super_mapping = {}
    for l in super_big_set:
        # cerebral GM
        # I've removed pallidum, putamen and caudate from deep gray matter (see below)
        if l in GIF_right_gray_matter | GIF_right_deep_gray_matter | GIF_right_amygdala | \
                GIF_left_gray_matter | GIF_left_deep_gray_matter | GIF_left_amygdala:

            v = _cerebral_GM

        # see https://github.com/vcasellesb/project_MARCOS/blob/ad729f68aa8d7a02d15d00012471cec3ce057b4d/marcos/generate_nnunet_dataset.py#L231
        elif l in GIF_optic_chiasm:
            v = _cerebral_GM

        # cerebral WM
        elif l in GIF_right_white_matter | GIF_left_white_matter | GIF_corpus_callosum:
            v = _cerebral_WM

        # cerebellar GM
        elif l in GIF_right_cerebellum_gray_matter | GIF_left_cerebellum_gray_matter | GIF_Cerebellar_Vermal_Lobules:
            v = _cerebellar_GM

        # cerebellar WM
        elif l in GIF_left_cerebellum_white_matter | GIF_right_cerebellum_white_matter:
            v = _cerebellar_WM

        # CSF separate ventricular and non-ventricular CSF
        elif l in GIF_right_ventricle | GIF_left_ventricle | GIF_third_ventricle | GIF_fourth_ventricle | GIF_fifth_ventricle:
            v = _ventricular_CSF
        elif l in GIF_non_ventricular_CSF:
            v = _outer_CSF

        # thalamus
        elif l in GIF_right_thalamus:
            v = _thalamus
        elif l in GIF_left_thalamus:
            v = _thalamus

        # ventral DC
        elif l in GIF_right_ventral_DC:
            v = _ventral_DC
        elif l in GIF_left_ventral_DC:
            v = _ventral_DC

        # caudate
        elif l in GIF_caudate:
            v = _caudate

        elif l in GIF_putamen | GIF_pallidum:
            v = _putamen_and_pallidum

        elif l in GIF_left_hippocampus | GIF_right_hippocampus:
            v = _hippocampus

        # brain stem and pons
        elif l in GIF_Brain_stem | GIF_pons:
            v = _brain_stem_and_pons

        # non brain
        elif l in GIF_Non_brain_low:
            v = _non_brain_low
        elif l in GIF_Non_brain_mid:
            v = _non_brain_mid
        elif l in GIF_Non_brain_high:
            v = _non_brain_high

        else:
            raise ValueError(f'Uncaught label at: {l = }')

        super_mapping[l] = v

    return super_mapping

MAPPING_FROM_GIF_TO_INTERNAL = construct_mapping_from_sets(GIF_all_labels)