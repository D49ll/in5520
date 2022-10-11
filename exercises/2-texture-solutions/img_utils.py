import matplotlib.pyplot as plt
import numpy as np


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
    # return np.uint8(np.digitize(img, np.arange(0, np.max(img),gray_levels)))


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
