import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from tqdm import tqdm

from img_utils import requantize
from img_utils import SlidingWindowIter

## CODE TAKEN FROM SOLUTION GIVEN TO EXERCISES ##

@jit(nopython=True)
def glcm(quantized, dy, dx, gray_levels=256, normalise=True, symmetric = True):
    glcm = np.zeros((gray_levels, gray_levels))
    for y in range(quantized.shape[0] - abs(dy)):
        for x in range(quantized.shape[1] - abs(dx)):
            px1 = quantized[y, x]
            px2 = quantized[y + dy, x + dx]
            glcm[px1, px2] += 1

    if normalise and symmetric:
        glcm += np.transpose(glcm) #Added this to the code

        return glcm * (1 / np.sum(glcm))

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


## CODE CREATED FOR THIS ASSIGNMENT ##

@jit(nopython=True)
def inertia(matrix):
    inr = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            inr += matrix[i, j] * (i - j)**2
    return inr

@jit (nopython=True)
def homogeneity(matrix):
    idm = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            idm += matrix[i, j] * (1 / (1+(i-j)**2))
    return idm

@jit (nopython = True)
def px(i, matrix):
    P = 0
    for j in range(matrix.shape[0]):
        P += matrix[i, j]
    return P

@jit (nopython = True)
def py(i, matrix):
    P = 0
    for j in range(matrix.shape[0]):
        P += matrix[j, i]
    return P

@jit (nopython = True)
def ux(matrix):
    u = 0
    for i in range(matrix.shape[0]):
        u += i * px(i, matrix)
    return u

@jit (nopython = True)
def uy(matrix):
    u = 0
    for i in range(matrix.shape[0]):
        u += i * py(i, matrix)
    return u

@jit (nopython = True)
def cluster_shade(matrix):
    u_x = ux(matrix)
    u_y = uy(matrix)

    shd = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shd += matrix[i, j] * (i+j-u_x-u_y)**3
    return shd

def isotropic_glcm(image, gray_levels, distance=1):
    quant_image = requantize(image, gray_levels=gray_levels)
    glcm_0 = glcm(quant_image, dx=distance, dy=0, gray_levels=gray_levels, normalise=True)
    glcm_45 = glcm(quant_image, dx=distance, dy=distance, gray_levels=gray_levels, normalise=True)
    glcm_90 = glcm(quant_image, dx=0, dy=distance, gray_levels=gray_levels, normalise=True)
    glcm_135 = glcm(quant_image, dx=-distance, dy=distance, gray_levels=gray_levels, normalise=True)

    isotropic = (glcm_0+glcm_45+glcm_90+glcm_135) / 4

    return isotropic


