"""
Tries to remove unnecessary black borders around the images, and
"trim" the images to they take up the entirety of the image.
"""

import os
import numpy as np
from PIL import Image
import cv2


"""
These following methods standardize the images withing one dataset
given that some of them are taken with different equipment.

The following methods are taken from the following github repository:
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Kaggles/DiabeticRetinopathy/preprocess_images.py

The changes made are realted to making the methods applicable to idividual images and not entire datasets.
Thus could be used as a pre-processing method for a single image.
"""

def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percentage of the pixels are above the
    threshold will be the first clipping point.

    Args:
        im {PIL.Image} -- Image to be trimmed

    Returns:
        PIL.Image -- Trimmed image
    """
    percentage = 0.02
    
    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)


def resize_maintain_aspect(image, desired_size):
    """
    Resizes an image to a desired size while maintaining the aspect ratio.
    Args:
        image {PIL.Image} -- Image to be resized
        desired_size {int} -- Desired size of the image
    Returns:
        PIL.Image -- Resized image
    """
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def pre_process_image(image_original, output_size):
    """
    Pre-processes an image by trimming i andresizing it, and converting it to an numpy array.
    Args:
        image_original {PIL.Image} -- Image to be pre-processed
        output_size {int} -- Desired size of the output image
    Returns:
        np.array -- Pre-processed image
    """
    image_trim = trim(image_original)
    image_resized = resize_maintain_aspect(image_trim, desired_size=output_size[0])
    return image_resized


# if __name__ == "__main__":
#     image_path = r"D:\Datasets\EyePACS\test\9_right.jpeg"
#     image = Image.open(image_path)
#     preprocessed_image = pre_process_image(image, output_size=(512, 512))

#     print(image.size)
#     print(preprocessed_image.size)