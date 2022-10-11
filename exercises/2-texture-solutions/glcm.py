from itertools import combinations_with_replacement

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from PIL import Image
from tqdm import tqdm

from img_utils import requantize
from img_utils import SlidingWindowIter
from img_utils import threshhold

# matplotlib.use("tkagg")


def contrast(matrix):
    return np.sum(
        [
            matrix[i, j] * i * j
            for i, j in combinations_with_replacement(
                range(matrix.shape[0]),
                2,
            )
        ],
    )


def entropy(matrix):
    eps = 1e-8  # avoids invalid value problems
    return np.sum(
        [
            -np.log(matrix[i, j] + eps) * ((matrix[i, j]) + eps)
            for i, j in combinations_with_replacement(
                range(matrix.shape[0]),
                2,
            )
        ],
    )


@jit(nopython=True)
def glcm(quantized, dy, dx, gray_levels=256, normalise=True):
    glcm = np.zeros((gray_levels, gray_levels))
    for y in range(quantized.shape[0] - abs(dy)):
        for x in range(quantized.shape[1] - abs(dx)):
            px1 = quantized[y, x]
            px2 = quantized[y + dy, x + dx]
            glcm[px1, px2] += 1

    if normalise:
        return glcm * (1 / np.sum(glcm))

    return glcm


def compute_glcm_features(image, window_size, dy, dx, gray_levels, feature_fn):
    feature_image = np.zeros(image.shape)
    quantized = requantize(image, gray_levels)

    for x, y, window in tqdm(SlidingWindowIter(quantized, window_size)):
        matrix = glcm(window, dy, dx, gray_levels)
        feature_image[y, x] = feature_fn(matrix)
    return feature_image


def test_glcm() -> None:
    # Example from lecture note week 2, page 34.
    expected = np.array(
        [[1, 2, 1, 0], [0, 1, 3, 0], [0, 0, 3, 5], [0, 0, 2, 2]],
    ) * (1 / 20)
    img = np.array(
        [
            [0, 1, 1, 2, 3],
            [0, 0, 2, 3, 3],
            [0, 1, 2, 2, 3],
            [1, 2, 3, 2, 2],
            [2, 2, 3, 3, 2],
        ],
    )

    res = glcm(img, dx=1, dy=0, gray_levels=4, normalise=True)
    np.testing.assert_equal(res, expected)
    res = glcm(img, dx=1, dy=1, gray_levels=4, normalise=True)
    assert res.min() == 0
    assert res.max() <= 1
    res = glcm(img, dx=0, dy=1, gray_levels=4, normalise=True)
    assert res.min() == 0
    assert res.max() <= 1
    res = glcm(img, dx=-3, dy=-3, gray_levels=4, normalise=False)
    assert res.min() == 0
    assert res.max() > 1
    res = glcm(img, dx=-3, dy=-3, gray_levels=4, normalise=True)
    assert res.min() == 0
    assert res.max() <= 1

    print("Tests passed")


def main():
    # Task 1
    test_glcm()

    image = np.asarray(Image.open("./zebra_3.tif").convert("L"))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title("Image")
    plt.axis("off")
    plt.show()
    plt.close()

    var_features = compute_glcm_features(image.copy(), 31, 0, 5, 8, np.var)
    plt.imshow(var_features, cmap=plt.cm.gray)
    plt.title("Variance feature map")
    plt.axis("off")
    plt.show()
    plt.close()

    entr_features = compute_glcm_features(image.copy(), 31, 0, 5, 8, entropy)
    plt.title("Entropy feature map")
    plt.imshow(entr_features, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
    plt.close()

    contrast_features = compute_glcm_features(
        image.copy(), 31, 0, 5, 8, contrast
    )
    plt.title("Contrast feature map")
    plt.imshow(contrast_features, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
    plt.close()

    # breakpoint()

    # Task 2 and 3 - see matlab solution

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
