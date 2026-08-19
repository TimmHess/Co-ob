CORRUPTIONS_SUBSET_MAP = {
    "all_sets": [   
        ["none"],
        ["impulse_noise", "gaussian_noise", "shot_noise", "speckle_noise"],
        ["defocus_blur", "gaussian_blur", "glass_blur", "motion_blur", "zoom_blur", "elastic_transform"],
        ["fog", "frost", "spatter", "snow"],
        ["contrast", "brightness", "jpeg_compression", "saturate"],
        ["pixelate"],
    ],
    "selection_separate":[
        "none",
        "impulse_noise",
        "motion_blur",
        "defocus_blur",
        #"glass_blur",
        "contrast",
        "brightness",
        "saturate",
        "jpeg_compression",
        "pixelate",
    ],
    "selection_small":[
        "none",
        "impulse_noise",
        "brightness",
        "jpeg_compression"
    ],
    "selection_small2":[
        "speckle_noise",
        "impulse_noise",
        "brightness",
        "contrast",
        "saturate",
        "jpeg_compression"
    ]
}

CORRUPTIONS_SEVERITY_MAP = {
    "selection_small":[
        1,
        1,
        3,
        5
    ],
    "selection_small2":[
        2,
        1,
        3,
        1,
        3,
        5
    ]
}


def get_num_experiences(corruption_set:str) -> (int, int):
    """
    Returns the number of training and evaluation experiences for a given corruption set.
    """
    if type(CORRUPTIONS_SUBSET_MAP[corruption_set][0]) == list:
        return len(CORRUPTIONS_SUBSET_MAP[corruption_set]), 1
    return len(CORRUPTIONS_SUBSET_MAP[corruption_set]), len(CORRUPTIONS_SUBSET_MAP[corruption_set])