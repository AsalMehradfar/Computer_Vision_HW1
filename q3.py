import numpy as np
import utils
import cv2
import read_config
import sift

path = read_config.path_config()
resource_path = path['resource_path']
res_path = path['res_path']

params = read_config.param_config('param_config_q3')
R = params['r']
MAX_ITERATIONS = params['max_iterations']


def automatic_sift(img1, img2, flag=False):
    """
    Finding match points between two images by RANSAC Algorithm

    Inputs:
    --> img1: the first desired image
    --> img: the second desired image
    --> flag: the default value is False, if it is True the outputs will be saved.
    Outputs:
    ==> Nothing, just plot outputs
    """

    kp1, kp2, good_matches = sift.sift(img1, img2, R, True)

    # draw all key points
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=[0, 255, 0])
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=[0, 255, 0])

    good_matches = sorted(good_matches, key=lambda x: x[0].distance)  # we can omit this line

    # draw good match points
    kp1_match = [kp1[m[0].queryIdx] for m in good_matches]
    img1_match = cv2.drawKeypoints(img1_kp, kp1_match, None, color=[0, 0, 255])
    kp2_match = [kp2[m[0].trainIdx] for m in good_matches]
    img2_match = cv2.drawKeypoints(img2_kp, kp2_match, None, color=[0, 0, 255])

    # draw good match lines
    match_lines = cv2.drawMatchesKnn(img1_match, kp1, img2_match, kp2, good_matches, None, flags=2,
                                     singlePointColor=(0, 0, 255), matchColor=(0, 0, 255))
    match_lines_final = cv2.drawMatchesKnn(img1_match, kp1, img2_match, kp2, good_matches[:20], None, flags=2,
                                           singlePointColor=(0, 0, 255), matchColor=(0, 0, 255))

    # apply RANSAC and find inliers
    src_pts = np.float32([kp.pt for kp in kp1_match]).reshape(-1, 1, 2)
    des_pts = np.float32([kp.pt for kp in kp2_match]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(src_pts, des_pts, cv2.RANSAC, maxIters=MAX_ITERATIONS)
    inlier = []
    for i in range(len(good_matches)):
        if status[i] == 1:
            inlier.append(good_matches[i])

    match_lines_no_points = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2,
                                               singlePointColor=(0, 0, 255), matchColor=(0, 0, 255))
    inlier_lines = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier, match_lines_no_points.copy(),
                                      flags=(cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + 2),
                                      singlePointColor=(255, 0, 0), matchColor=(255, 0, 0))
    # inlier_lines = cv2.drawMatchesKnn(img1, kp1, img2, kp2, inlier, None, flags = 2,
    #                                   singlePointColor = (255, 0, 0), matchColor= (255, 0, 0))   # just inliers

    # homography
    img_homography = sift.perspective(img2, np.linalg.inv(H))

    if flag:
        utils.plot_two_imgs(img1_kp, img2_kp, res_path + 'res13_corners.jpg')
        utils.plot_two_imgs(img1_match, img2_match, res_path + 'res14_correspondences.jpg')
        utils.plot_img(match_lines, res_path + 'res15_matches.jpg')
        utils.plot_img(match_lines_final, res_path + 'res16.jpg')
        utils.plot_img(inlier_lines, res_path + 'res17.jpg')
        utils.plot_img(img_homography, res_path + 'res19.jpg')
    else:
        utils.plot_two_imgs(img1_kp, img2_kp)
        utils.plot_two_imgs(img1_match, img2_match)
        utils.plot_img(match_lines)
        utils.plot_img(match_lines_final)
        utils.plot_img(inlier_lines)
        utils.plot_img(img_homography)


img1 = utils.get_img(resource_path + 'im03.jpg')
img2 = utils.get_img(resource_path + 'im04.jpg')
automatic_sift(img1, img2, True)
