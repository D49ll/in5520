import numpy as np
from seaborn import heatmap
import matplotlib.pyplot as plt
from PIL import Image




class SlidingWindowIter:
    def __init__(self, image, window_size):
        self.wsize = window_size
        self.x = 0
        self.y = 0
        self.count = 0
        self.pad = (window_size - 1) // 2
        self.img_dims = image.shape
        self.padded = np.pad(
            image,
            ((self.pad, self.pad), (self.pad, self.pad)),
            mode="reflect",
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self.y >= self.img_dims[0] + self.pad - 1:
            raise StopIteration
        else:
            self.x = (self.count) % self.img_dims[1] + self.pad
            self.y = self.count // self.img_dims[0] + self.pad
            self.count += 1
            return (
                self.x - self.pad,
                self.y - self.pad,
                self.padded[
                    self.y - self.pad : self.y + self.pad + 1,
                    self.x - self.pad : self.x + self.pad + 1,
                ],
            )
        # else:
        #     raise StopIteration


def requantize(img, gray_levels):
    return np.uint8(np.round(img * ((gray_levels - 1) / 255)))


def threshhold(img, T, reverse=False):
    """
    Simple global threshhold
    """

    thresholded = np.zeros(img.shape, dtype=np.uint8)
    for index, val in np.ndenumerate(img):
        if not reverse:
            if val < T:
                thresholded[index[0], index[1]] = 1
        else:
            if val > T:
                thresholded[index[0], index[1]] = 1
    return thresholded

def save_image(image, glcm_image, gray_levels, dx, dy, filename, texture_start):
    fig = plt.figure(figsize=(20,10))
    ax1, ax2 = fig.subplots(2,4)   
    
    for i in range(4):
        ax1[i].axis('off')
        ax1[i].set_aspect(1)

        ax2[i].axis('off')
        ax2[i].set_aspect(1)

    ax1[0].imshow(image[0], cmap=plt.cm.gray)
    ax1[0].set_title(f'texture {texture_start+1}')

    heatmap(glcm_image[0],ax=ax1[1])
    ax1[1].set_title(f'dx: {dx[0]}, dy: {dy[0]}')

    ax1[2].imshow(image[1], cmap=plt.cm.gray)
    ax1[2].set_title(f'texture {texture_start+2}')

    heatmap(glcm_image[1],ax=ax1[3])
    ax1[3].set_title(f'dx: {dx[1]}, dy: {dy[1]}')

    ax2[0].imshow(image[2],cmap=plt.cm.gray)
    ax2[0].set_title(f'texture {texture_start+3}')

    heatmap(glcm_image[2],ax=ax2[1])
    ax2[1].set_title(f'dx: {dx[2]}, dy: {dy[2]}')
        
    ax2[2].imshow(image[3],cmap=plt.cm.gray)
    ax2[2].set_title(f'texture {texture_start+4}')

    heatmap(glcm_image[3],ax=ax2[3])
    ax2[3].set_title(f'dx: {dx[3]}, dy: {dy[3]}')

    fig.suptitle(f'GLCM of {filename} with gray_levels={gray_levels}')
    plt.savefig(f"{filename}_glcms.png") 



def read_image(filename):
    f = np.asarray(Image.open(filename))
    return f

def sub_images(image):
    return np.array([image[:256, :256], image[:256,256:], image[256:,:256], image[256:,256:]],dtype=np.uint8)

def save_images(images,filename):
    for i in range(len(images)):
        Image.fromarray(images[i]).save(filename+"_texture"+str(i+1)+".png")

def extract_subimages(filename):
    img = read_image(filename+'.png')
    sub = sub_images(img)
    return sub