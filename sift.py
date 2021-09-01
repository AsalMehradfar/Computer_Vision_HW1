import cv2
import numpy as np
import utils


def sift(img1, img2, R, flag=False):
    """
    Finding match points between two images by RANSAC Algorithm

    Inputs:
    --> img1: the first desired image
    --> img2: the second desired image
    --> R: the accepted ratio for distance between matches
    -->
    Outputs:
    ==> kp1: key points of the first image
    ==> kp2: key points of the second image
    ==> good_matches: good matches of two images
    """

    sift_cr = cv2.SIFT_create()

    # find all key points
    kp1, des1 = sift_cr.detectAndCompute(img1, None)
    kp2, des2 = sift_cr.detectAndCompute(img2, None)

    # find all matches
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < R * n.distance:
            if flag:
                good_matches.append([m])
            else:
                good_matches.append(m)

    return kp1, kp2, good_matches


def perspective(img, H):
    """
    computing the homography of the image with matrix H

    Inputs:
    --> img: the desired image
    --> H: the homography matrix
    Outputs:
    ==> img_homography: the output image
    """
    x_min, y_min, x_max, y_max = corners_homography(img, H)
    [x_offset, y_offset] = [-x_min, -y_min]
    [x_size, y_size] = [x_max - x_min, y_max - y_min]
    offset = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]])
    img_homography = cv2.warpPerspective(img, np.dot(offset, H), (x_size, y_size))
    return img_homography


def corners_homography(img, H):
    """
    computing the min and max of the corners of an image
    after applying the homography matrix without offset
    we use these points to find offset and size of the homography

    Inputs:
    --> img: the desired image
    --> H: the homography matrix
    Outputs:
    ==> x_min: minimum x after homography
    ==> y_min: minimum y after homography
    ==> x_max: maximum x after homography
    ==> y_max: maximum y after homography
    """
    height, width, _ = img.shape
    corners = [np.array([[0, 0, 1]]).transpose(),
               np.array([[width - 1, 0, 1]]).transpose(),
               np.array([[0, height - 1, 1]]).transpose(),
               np.array([[width - 1, height - 1, 1]]).transpose()]
    [x_min, y_min, x_max, y_max] = [-1 for _ in range(4)]
    for c in corners:
        m = np.matmul(H, c)
        if x_min == -1:
            [x_min, y_min, x_max, y_max] = [int(m[0] / m[2]),
                                            int(m[1] / m[2]),
                                            int(m[0] / m[2]),
                                            int(m[1] / m[2])]
        else:
            [x_min, y_min, x_max, y_max] = [min(x_min, int(m[0] / m[2])),
                                            min(y_min, int(m[1] / m[2])),
                                            max(x_max, int(m[0] / m[2])),
                                            max(y_max, int(m[1] / m[2]))]
    return x_min, y_min, x_max, y_max


def plot_inliers(status, kp1, kp2, matches, img1, img2):
    """
    plotting the inliers, THIS FUNCTION WAS NOT WANTED IN THE SHEET!

    Inputs:
    --> H: the best homography matrix
    --> status: the status matrix for matches, 1 means the match is inlier and 0 zero means the match is outlier
    --> kp1: key points of the first image
    --> kp2: key points of the second image
    --> matches: good matches of two images
    --> img1: the first desired image
    --> img2: the second desired image
    Outputs:
    ==> Nothing, just plotting the inliers
    """
    inlier = []
    for i in range(len(matches)):
        if status[i] == 1:
            inlier.append(matches[i])
    # print(len(inlier))
    inlier_lines = cv2.drawMatches(img1, kp1, img2, kp2, inlier, None, flags=2, singlePointColor=(255, 0, 0),
                                   matchColor=(255, 0, 0))
    utils.plot_img(inlier_lines)
