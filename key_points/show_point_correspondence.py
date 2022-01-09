#
# Author: Denis Tananaev
# Date: 09.01.2022
#

import cv2
import numpy as np
import matplotlib.pyplot as plt

from helpers.corner_detector  import compute_corners
from helpers.descriptor import compute_descriptors, compute_matches
from PIL import Image
import imageio
def plot_matches(I1, I2, C1, C2, M):
    """ 
    Plots the matches between the two images
    """
    # Create a new image with containing both images
    W = I1.shape[1] + I2.shape[1]
    H = np.max([I1.shape[0], I2.shape[0]])
    I_new = np.zeros((H, W), dtype=np.uint8)
    I_new[0:I1.shape[0], 0:I1.shape[1]] = I1
    I_new[0:I2.shape[0], I1.shape[1]:I1.shape[1] + I2.shape[1]] = I2

    # plot matches
    plt.imshow(I_new, cmap='gray')
    for i in range(M.shape[0]):
        p1 = C1[M[i, 0], :]
        p2 = C2[M[i, 1], :] + np.array([I1.shape[1], 0])
        plt.plot(p1[0], p1[1], 'rx')
        plt.plot(p2[0], p2[1], 'rx')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-y')


if __name__ == "__main__":
    T_harris = 250
    T_shi_tomasi = 110
    wtf= np.asarray(imageio.imread('mountain_1.jpg'))
    image_1 = np.asarray(Image.open("mountain_1.jpg"))
    image_2 = np.asarray(Image.open("mountain_2.jpg"))
    print(f"wtf {wtf.shape} image_1 {image_1.shape}")
    I1_gray = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    I2_gray = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
    # compute corner keypoints
    C1_harris =  compute_corners(I1_gray, 'harris', T_harris)
    C2_harris =  compute_corners(I2_gray, 'harris', T_harris)


    C1_shi_tomasi =  compute_corners(I1_gray, 'shi-tomasi', T_shi_tomasi)
    C2_shi_tomasi =  compute_corners(I2_gray, 'shi-tomasi', T_shi_tomasi)

    # compute the descriptor for the two images 
    print(f"compute_descriptors")
    D1_harris = compute_descriptors(I1_gray, C1_harris)
    D2_harris = compute_descriptors(I2_gray, C2_harris)

    D1_shi_tomasi = compute_descriptors(I1_gray, C1_shi_tomasi)
    D2_shi_tomasi = compute_descriptors(I2_gray, C2_shi_tomasi)
    print(f"D1_harris {D1_harris.shape}, D2_harris {D2_harris.shape}")
    print(f"D1_shi_tomasi {D1_shi_tomasi.shape}, D2_shi_tomasi {D2_shi_tomasi.shape}")
    print(f"compute matches")
    # Matches for harris corner keypoints and corresponding keypoints
    M_harris = compute_matches(D1_harris, D2_harris)

    # Matches for Shi-Tomasi keypoints and corresponding keypoints
    M_shi_tomasi = compute_matches(D1_shi_tomasi, D2_shi_tomasi)
    print(f"M_harris {M_harris.shape}, M_shi_tomasi {M_shi_tomasi.shape}")


    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(I1_gray, cmap='gray')
    plt.plot(C1_harris[:,1], C1_harris[:, 0], 'rx', markersize=12)
    plt.xlabel("Harris Corners I1")

    plt.subplot(1,2,2)
    plt.imshow(I2_gray, cmap='gray')
    plt.plot(C2_harris[:,1], C2_harris[:,0], 'rx', markersize=12)
    plt.xlabel("Harris Corners I2")

    C1_harris_transpose = np.zeros_like(C1_harris)
    C1_harris_transpose[:, 0], C1_harris_transpose[:, 1] = C1_harris[:, 1], C1_harris[:, 0]

    C2_harris_transpose = np.zeros_like(C2_harris)
    C2_harris_transpose[:, 0], C2_harris_transpose[:, 1] = C2_harris[:, 1], C2_harris[:, 0]


    C1_shi_tomasi_transpose = np.zeros_like(C1_shi_tomasi)
    C1_shi_tomasi_transpose[:, 0], C1_shi_tomasi_transpose[:, 1] = C1_shi_tomasi[:, 1], C1_shi_tomasi[:, 0]

    C2_shi_tomasi_transpose = np.zeros_like(C2_shi_tomasi)
    C2_shi_tomasi_transpose[:, 0], C2_shi_tomasi_transpose[:, 1] = C2_shi_tomasi[:, 1], C2_shi_tomasi[:, 0]
    # Visualize the matches
    plt.figure()
    plot_matches(I1_gray, I2_gray, C1_harris_transpose, C2_harris_transpose, M_harris)

    plt.figure()
    plot_matches(I1_gray, I2_gray, C1_shi_tomasi_transpose, C2_shi_tomasi_transpose, M_shi_tomasi)
    plt.show()