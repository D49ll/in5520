from glcm import glcm, compute_glcm_features, homogeneity, inertia, cluster_shade, isotropic_glcm
from img_utils import requantize, threshhold, save_image, extract_subimages, read_image
import matplotlib.pyplot as plt


def taskB():
    t1 = extract_subimages('mosaic1')
    t2 = extract_subimages('mosaic2')

    #Gray levels
    gray_levels = 16

    #Mosaic1 parameters
    glcm_mosiac1 = []
    dx = [1, 4,           2, 4]
    dy = [1, "isotropic", 0, "isotropic"]

    #Mosaic1 parameters
    glcm_mosiac2 = []
    dx_2 = [1, 0, 2, 4]
    dy_2 = [1, 2, 0, "isotropic"]

    #Mosaic1
    glcm_mosiac1.append(glcm(requantize(t1[0], gray_levels=gray_levels), dx=dx[0], dy=dy[0], gray_levels=gray_levels, normalise=True))
    glcm_mosiac1.append(isotropic_glcm(t1[1], gray_levels, distance = dx[1]))
    glcm_mosiac1.append(glcm(requantize(t1[2], gray_levels=gray_levels), dx=dx[2], dy=dy[2], gray_levels=gray_levels, normalise=True))
    glcm_mosiac1.append(isotropic_glcm(t1[3], gray_levels, distance = dx[3]))
    save_image(t1, glcm_mosiac1, gray_levels=gray_levels, dx=dx, dy=dy, filename='mosaic1', texture_start=0)

    #Mosaic2
    glcm_mosiac2.append(glcm(requantize(t2[0], gray_levels=gray_levels), dx=dx_2[0], dy=dy_2[0], gray_levels=gray_levels, normalise=True))
    glcm_mosiac2.append(glcm(requantize(t2[1], gray_levels=gray_levels), dx=dx_2[1], dy=dy_2[1], gray_levels=gray_levels, normalise=True))
    glcm_mosiac2.append(glcm(requantize(t2[2], gray_levels=gray_levels), dx=dx_2[2], dy=dy_2[2], gray_levels=gray_levels, normalise=True))
    glcm_mosiac2.append(isotropic_glcm(t2[3], gray_levels, distance = dx_2[3]))
    
    save_image(t2, glcm_mosiac2, gray_levels=gray_levels, dx=dx_2, dy=dy_2, filename='mosaic2', texture_start=4)


def taskC_D(filename, window_size, dx, dy, gray_levels, threshold):
    img1 = read_image(filename)

    homogen = compute_glcm_features(img1.copy(), window_size=window_size, dx=dx, dy=dy, gray_levels=gray_levels, feature_fn=homogeneity)
    inert = compute_glcm_features(img1.copy(), window_size=window_size, dx=dx, dy=dy, gray_levels=gray_levels, feature_fn=inertia)
    clust = compute_glcm_features(img1.copy(), window_size=window_size, dx=dx, dy=dy, gray_levels=gray_levels, feature_fn=cluster_shade)

    homogen_T = threshhold(homogen, threshold[0])
    inert_T = threshhold(inert, threshold[1])
    clust_T = threshhold(clust, threshold[2])

    fig = plt.figure(figsize=(20,10))
    ax1, ax2, ax3 = fig.subplots(3,3)
    
    for i in range(2):
        ax1[i].axis('off')
        ax1[i].set_aspect(1)

        ax2[i].axis('off')
        ax2[i].set_aspect(1)

        ax3[i].axis('off')
        ax3[i].set_aspect(1)        

    ax1[0].imshow(img1, cmap=plt.cm.gray)
    ax1[0].set_title(f'{filename}')

    ax1[1].imshow(homogen, cmap=plt.cm.gray)
    ax1[1].set_title(f'homogeneity')

    ax1[2].imshow(homogen_T, cmap=plt.cm.gray)
    ax1[2].set_title(f'homogeneity t={threshold[0]}')

    ax2[0].imshow(img1, cmap=plt.cm.gray)
    ax2[0].set_title(f'{filename}')

    ax2[1].imshow(inert, cmap=plt.cm.gray)
    ax2[1].set_title(f'inertia')

    ax2[2].imshow(inert_T, cmap=plt.cm.gray)
    ax2[2].set_title(f'inert t={threshold[1]}')

    ax3[0].imshow(img1, cmap=plt.cm.gray)
    ax3[0].set_title(f'{filename}')

    ax3[1].imshow(clust, cmap=plt.cm.gray)
    ax3[1].set_title(f'cluster_shade')

    ax3[2].imshow(clust_T, cmap=plt.cm.gray)
    ax3[2].set_title(f'cluster_shade t={threshold[2]}')

    fig.suptitle(f'Feature maps from {filename} with gray_levels={gray_levels} and dx={dx}, dy={dy}')
    plt.savefig(f"{filename}_featuremaps_dx{dx}_dy{dy}.png") 

    # plt.show()

def main():
    taskB()
    taskC_D('mosaic1.png', window_size=31, dx=5, dy=0, gray_levels=16, threshold=[0.28,20,0])
    taskC_D('mosaic2.png', window_size=31, dx=0, dy=5, gray_levels=16, threshold=[0.4,10,0])




main()
