import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np


def plot_img(img, path=None):
    """
    Plot a colorful image and save it if needed

    Inputs:
    --> img: the desired image
    --> path: the default value is None, if it is given the image will be saved in the path
    Outputs:
    ==> Nothing, the image will be plotted
    """
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.axis('off')
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.show()


def get_img(path):
    """
    Read the image file from the path and change it from BGR to RGB
    pay attention that in open-cv colorful images are BGR **NOT** RGB

    Inputs:
    --> path: path for the image
    Outputs:
    ==> img: the RGB image
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def plot_gray_img(img, path=None):
    """
    Plot a gray img and save it if needed

    Inputs:
    --> img: the desired image
    --> path: the default value is None, if it is given the image will be saved in the path
    Outputs:
    ==> Nothing, the image will be plotted
    """
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.show()


def get_gray_img(path):
    """
    Read the image file from the path and change it to the gray one

    Inputs:
    --> path: path for the image
    Outputs:
    ==> gray_img: the gray image
    """
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def plot_two_imgs(img1, img2, path=None):
    """
    Plot two colorful images with subplot and save it if needed

    Inputs:
    --> img1: the first desired image
    --> img2: the second desired image
    --> path: the default value is None, if it is given the image will be saved in the path
    Outputs:
    ==> Nothing, the image will be plotted
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(img1)
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].axis('off')
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.show()


def save_img(array, path, scaled=False):
    """
    save the input image in the desired path

    Inputs:
    --> array: the array of an image
    --> path: the desired path for saving the image
    --> scaled: the default value is False,
    if it is given True the image will be scaled into [0,255]
    Outputs:
    ==> Nothing, just saving the image
    """
    if scaled:
        array = scaling_img(array).astype(np.uint8)
    img = Image.fromarray(array)
    img.save(path)


def scaling_img(img):
    """
    return the scaled image usually for saving

    Inputs:
    --> img: the desired image for scaling
    Outputs:
    ==> scaled_img: we assume that the minimum of the image is zero
        so we scale it by division by its maximum and multiplying it by 255.0
    """
    scaled_img = 255.0 * img / np.max(img)
    return scaled_img


def normalize_img(img):
    """
    return the normalized image

    Inputs:
    --> img: the desired image for scaling
    Outputs:
    ==> norm_img: we assume that the minimum of the image is zero
        so we normalize it by division by its maximum
    """
    norm_img = img / np.max(img)
    return norm_img
