import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, iterate_structure


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    struct = generate_binary_structure(2, 1)
    struct = iterate_structure(struct, radius)
    return binary_closing(mask, structure=struct)


def postprocess(mask: np.ndarray, fill: bool, dilate: int) -> np.ndarray:
    if fill:
        mask = fill_holes(mask)
    if dilate:
        mask = close_mask(mask, dilate)
    return mask
