import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def requantize(img, gray_levels):
    return np.round(img * (gray_levels-1)/255)

def equalize_histogram(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img.astype('uint8')).astype('float')

@jit(nopython=True)
def compute_glcm(img, dx, dy, gray_levels, norm):
    n, m = img.shape
    glcm = np.zeros((gray_levels, gray_levels))
    for x in range(n - abs(dx)):
        for y in range(m - abs(dy)):
            px0 = int(img[x, y])
            px1 = int(img[x + dx, y + dy])
            glcm[px0, px1] += 1

    sum = np.sum(glcm)
    if norm and sum != 0:
        return glcm * (1 / sum)

    return glcm


#_____________GLCM_FEATURES_____________#

@jit(nopython=True)
def homogeneity(glcm):
    n, m = glcm.shape
    sum = 0
    for i in range(n):
        for j in range(m):
            sum += glcm[i,j] / (1 + (i - j)**2)
    return sum

@jit(nopython=True)
def inertia(glcm):
    n, m = glcm.shapep
    sum = 0
    for i in range(n):
        for j in range(m):
            sum += glcm[i,j] * (i - j)**2
    return sum

@jit(nopython=True)
def cluster_shade(glcm):
    n, m = glcm.shape
    sum = 0
    px = np.sum(glcm, axis=1)/n
    py = np.sum(glcm, axis=0)/m
    for i in range(n):
        for j in range(m):
            sum += glcm[i,j] * (i + j - i*px[i] - j*py[j])**3
    return sum

#_____________END_FEATURES_____________#

@jit(nopython=True)
def feature_image(img, w, dx, dy, gray_levels, fn, norm):
    n, m = img.shape
    out = np.zeros((n-w*2, m-w*2))
    for i in range(w, n-w):
        for j in range(w, m-w):
            glcm = compute_glcm(img[i-w:i+w, j-w:j+w], dx, dy, gray_levels, norm)
            out[i-w,j-w] = fn(glcm)
    return out

@jit(nopython=True)
def threshold(feat, t):
    n, m = feat.shape
    thresh = np.max(feat) * t
    out = np.zeros(feat.shape)
    for i in range(n):
        for j in range(m):
            if feat[i,j] > thresh:
                out[i,j] = 1
    return out

def otsu_threshold(feat):
    feat = (feat*255).astype('uint8')
    _, out =  cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out

def plot_glcm(imgs, ds, gray_levels=16, norm=True, eq_hist=False):

    for i, img in enumerate(imgs):

        if eq_hist:
            img = equalize_histogram(img)

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes.flat[0].imshow(img, cmap='gray')
        axes.flat[0].set_title('Texture')

        img_quant = requantize(img, gray_levels)
        img_glcm = compute_glcm(img_quant, ds[i][0], ds[i][1], gray_levels, norm)
        im = axes.flat[1].imshow(img_glcm)
        axes.flat[1].set_title(f'dx={ds[i][0]}, dy={ds[i][1]}')

        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.5, hspace=0.5)
        
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        fig.colorbar(im, cax=cb_ax)

        plt.savefig(f'output_images/glcm_texture{i}.png')
        plt.close()

def plot_feature_images(imgs, ds, gray_levels=16, norm=True, w=15, ts=[0.5, 0.5], fn=homogeneity, pad='reflect', eq_hist=False, otsu=False):

    for i, img in enumerate(imgs):

        if eq_hist:
            img = equalize_histogram(img)

        dx, dy = ds[i]
        t = ts[i]
        img_quant = requantize(img, gray_levels)
        img_pad = np.pad(img_quant, w, pad)
        img_feat = feature_image(img_pad, w, dx, dy, gray_levels, fn, norm)
        if otsu:
            img_thresh = otsu_threshold(img_feat)
        else:
            img_thresh = threshold(img_feat, t)

        _, axes = plt.subplots(nrows=2, ncols=2)
        axes.flat[0].imshow(img, cmap='gray')
        axes.flat[1].imshow(img_pad, cmap='gray')
        axes.flat[2].imshow(img_feat, cmap='gray')
        axes.flat[3].imshow(img_thresh, cmap='gray')

        fn_str = fn.__name__
        plt.savefig(f'output_images/{fn_str}_img{i}.png')
        plt.close()

def main():

    #Creating images and lists
    img1 = cv2.imread('mosaic1.png', 0).astype('float')
    img2 = cv2.imread('mosaic2.png', 0).astype('float')
    imgs = [img1, img2]
    textures = []
    for img in imgs:
        textures.append(img[:int(img.shape[0]/2),:int(img.shape[1]/2)])
        textures.append(img[:int(img.shape[0]/2),int(img.shape[1]/2):])
        textures.append(img[int(img.shape[0]/2):,:int(img.shape[1]/2)])
        textures.append(img[int(img.shape[0]/2):,int(img.shape[1]/2):])

    #Parameters for GLCM plotting
    ds = [(1,1), (0,1),        
          (1,0), (1,1),         #dx and dy parameters in tuple
          (1,1), (0,1),         #for the eight textures
          (1,0), (1,1)]         
    gray_levels = 16            #Gray levels for quantizing
    norm = True                 #Normalize GLCM
    eq_hist = True              #Adaptive equalization of histogram

    plot_glcm(textures, ds, gray_levels, norm, eq_hist)

    #Parameters for feature images
    fn = homogeneity            #Feature to plot
    pad = 'reflect'             #Padding method for feature image
    w = 31                      #Window size for feature image
    ds = [(1,0), (0,1)]         #Single parameters for feature images
    gray_levels = 16            #Gray levels for quantizing
    norm = True                 #Normalize GLCM
    eq_hist = True              #Adaptive equalization of histogram
    ts = [0.7, 0.8]             #Threshold for feature image
    otsu = False                #Enabling Otsu disables threshold paramter above

    plot_feature_images(imgs, ds, gray_levels, norm, w, ts, fn, pad, eq_hist, otsu)

if __name__ == '__main__':
    main()