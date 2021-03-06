import numpy as np
import utils
import sift
import read_config

path = read_config.path_config()
resource_path = path['resource_path']
res_path = path['res_path']

params = read_config.param_config('param_config_q4')
R = params['r']
MAX_ITERATIONS = params['max_iterations']
INLIER_THRESHOLD = params['inlier_threshold']


def get_h(src_pts, des_pts):
    """
    Finding the homography matrix between 4 points

    Inputs:
    --> src_pts: a 4*1*2 array of 4 source points
    --> des_pts: a 4*1*2 array of 4 destination points
    Outputs:
    ==> h: the homography matrix between 4 points
    """
    n = len(src_pts)
    A = np.zeros((2 * n, 9))
    i = 0

    for s, d in zip(src_pts, des_pts):
        [x, y] = s[0]
        [xp, yp] = d[0]
        A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * yp, y * yp, yp]
        A[2 * i + 1, :] = [0, 0, 0, -x, -y, -1, x * xp, y * xp, xp]
        i += 1

    _, _, Vt = np.linalg.svd(A)

    h = Vt[-1, :].reshape(3, 3)
    h[[0, 1]] = h[[1, 0]]
    return h


def get_pts_mat(kp1, kp2, matches):
    """
    Making two matrixes of source and destination points
    with regard to the third axis, z

    Inputs:
    --> kp1: key points of the first image
    --> kp2: key points of the second image
    --> matches: good matches of two images
    Outputs:
    ==> src_mat: the matrix of source points with z axis defined as 1
    ==> des_mat: the matrix of destination points with z axis defined as 1
    """
    src_mat = np.zeros((3, len(matches)))
    des_mat = np.zeros((3, len(matches)))
    i = 0

    for m in matches:
        src_mat[:, i] = [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1]
        des_mat[:, i] = [kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1], 1]
        i += 1

    return src_mat, des_mat


def get_error(src_mat, des_mat, h):
    """
    Making a vector of errors generated by homography matrix h

    Inputs:
    --> src_mat: the matrix of source points with z axis defined as 1
    --> des_mat: the matrix of destination points with z axis defined as 1
    --> h: the homography matrix
    Outputs:
    ==> err: the vector of errors, for each set of source and destination points we have an special error
    """
    expected_des_mat = np.dot(h, src_mat)
    expected_des_mat = expected_des_mat / expected_des_mat[2, :]
    err = np.sqrt(np.sum((des_mat - expected_des_mat) ** 2, 0))
    return err


def find_homography(kp1, kp2, matches):
    """
    Making the final homography matrix and also the status of each matched key point

    Inputs:
    --> kp1: key points of the first image
    --> kp2: key points of the second image
    --> matches: good matches of two images
    Outputs:
    ==> H: the best homography matrix
    ==> status: the status matrix for matches, 1 means the match is inlier and 0 zero means the match is outlier
    """

    src_mat, des_mat = get_pts_mat(kp1, kp2, matches)
    n = len(matches)
    iter = 0
    final_match = 0

    while iter < MAX_ITERATIONS:
        random_idxs = np.random.choice(n, size=4, replace=False)
        src_pts = np.float32([src_mat[:2, i] for i in random_idxs]).reshape(-1, 1, 2)
        des_pts = np.float32([des_mat[:2, i] for i in random_idxs]).reshape(-1, 1, 2)

        h = get_h(src_pts, des_pts)
        err = get_error(src_mat, des_mat, h)

        stat = [1 if err[i] < INLIER_THRESHOLD else 0 for i in range(len(matches))]
        num_of_match = np.sum(stat)
        # print(num_of_match)

        if num_of_match > final_match:
            final_match = num_of_match
            status = stat
            H = h

        iter += 1

    return H, status


img1 = utils.get_img(resource_path + 'im03.jpg')
img2 = utils.get_img(resource_path + 'im04.jpg')
kp1, kp2, matches = sift.sift(img1, img2, R)
H, status = find_homography(kp1, kp2, matches)
# sift.plot_inliers(status, kp1, kp2, matches, img1, img2)
img_homography = sift.perspective(img2, np.linalg.inv(H))
utils.plot_img(img_homography, res_path + 'res20.jpg')
