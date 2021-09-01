import numpy as np
import math
import utils
import read_config

path = read_config.path_config()
resource_path = path['resource_path']
res_path = path['res_path']

params = read_config.param_config('param_config_q2')
D = params['d']
H = params['h']
N = params['n']

n = np.array([[0, 0, -1]])


def get_params(f, Px, Py, d, h, s=0):
    """
    compute the parameters needed for making K, K^(-1), R, t

    Inputs:
    --> f: focal length or number of camera pixels
    --> Px: x center of image
    --> Py: y center of image
    --> d: distance between camera and center of the football field
    --> h: height of the camera from the earth
    --> s: is usually zero or near zero for creating K
    Outputs:
    ==> K: here is equal to K' in the formula
    ==> K_inv: K^(-1)
    ==> R
    ==> t
    """
    K = np.array([[f, s, Px], [0, f, Py], [0, 0, 1]])
    K_inv = np.linalg.inv(K)
    angle = -np.arctan(d / h)
    R = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    C = np.array([[d, 0, 0]]).transpose()
    t = -np.dot(R, C)
    return K, K_inv, R, t


def get_H(K, K_inv, R, t, n, d):
    """
    compute H and H_inv

    Inputs:
    --> K: 3*3 array, here is equal to K' in the formula
    --> K_inv: K^(-1)
    --> R: 3*3 array
    --> t: 3*1 vector
    --> n: a vector which is vertical to the plane, being negative was understood by experiment
    --> d: distance between camera and center of the football field
    Outputs:
    ==> H: the homography 3*3 array computed by the formula in page 15 of slide 8
    ==> H_inv: H^(-1)
    """
    H = np.dot(np.dot(K, R - np.dot(t, n) / d), K_inv)
    H_inv = np.linalg.inv(H)
    return H, H_inv


def make_new_img(H, H_inv, img):
    """
    here at first we try to find the size of the output image
    by computing the effect of H inverse on the points in the input image
    and finding their maximum and minimum.
    then by making a completely zero 3d array of the new size we use H
    for computing new coordinates in the original image and used the values of the original image
    for setting the values of new image.

    Inputs:
    --> H: the homography 3*3 array computed by the formula in page 15 of slide 8
    --> H_inv: H^(-1)
    --> img: the original image, here the original logo
    Outputs:
    ==> new_img: the output image after homography
    """
    min_x = math.inf
    min_y = math.inf
    max_x = 0
    max_y = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            a = np.array([[x, y, 1]]).transpose()
            b = np.dot(H_inv, a)
            [x_new, y_new] = [int(b[0] / b[2]), int(b[1] / b[2])]
            min_x = min(x_new, min_x)
            min_y = min(y_new, min_y)
            max_x = max(x_new, max_x)
            max_y = max(y_new, max_y)

    new_img = np.zeros((max_x - min_x + 1, max_y - min_y + 1, 3))
    for x in range(new_img.shape[0]):
        for y in range(new_img.shape[1]):
            a = np.array([[x + min_x, y + min_y, 1]]).transpose()
            b = np.dot(H, a)
            [x_new, y_new] = [int(b[0] / b[2]), int(b[1] / b[2])]
            if 0 < x_new < img.shape[0] and 0 < y_new < img.shape[1]:
                new_img[x, y, :] = img[x_new, y_new, :]
    new_img = new_img.astype(np.uint8)
    return new_img


img = utils.get_img(resource_path + 'logo.png')
K, K_inv, R, t = get_params(N, img.shape[0] / 2, img.shape[0] / 2, D, H)
H, H_inv = get_H(K, K_inv, R, t, n, H)
new_img = make_new_img(H, H_inv, img)
utils.save_img(new_img, res_path + 'res12.jpg')
