import numpy as np
import cv2



# Calculate the geometric distance between estimated points and original points, namely residuals.
def compute_residual(P1, P2, H):
    """
    Compute the residual given the Homography H
    
    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2
    H: [numpy array 3x3]
        Homography which maps P1 to P2
    
    Returns:
    ----------
    residual : float
                residual computed for the corresponding points P1 and P2 
                under the transformation given by H
    """
    H = np.asarray(H)
    P1_hom = np.concatenate((P1, np.ones_like(P1[:, 0])[..., None]), axis=-1)
    P2_hom = np.concatenate((P2, np.ones_like(P2[:, 0])[..., None]), axis=-1)
    P2_proj = np.dot(H, P1_hom.T).T

    P1_proj = np.dot(np.linalg.inv(H), P2_hom.T).T

    P1_proj = P1_proj[:, :2] / P1_proj[:, 2][:, None]
    P2_proj = P2_proj[:, :2] / P2_proj[:, 2][:, None]
    residual = np.linalg.norm(P1-P1_proj, axis=-1) + np.linalg.norm(P2-P2_proj, axis=-1)


    return residual


def calculate_homography_four_matches(P1, P2):
    """
    Estimate the homography given four correspondening keypoints in the two images.

    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2

    Returns:
    ----------
    H: [numpy array 3x3]
        Homography which maps P1 to P2 based on the four corresponding points
    """
    if P1.shape[0] != 4 or P2.shape[0] != 4:
        print('Four corresponding points needed to compute Homography')
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)
    A = []
    for i in range(P1.shape[0]):
        p1 = np.array([P1[i, 0], P1[i, 1], 1])
        p2 = np.array([P2[i, 0], P2[i, 1], 1])

        a2 = [
            0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
            p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]
        ]
        a1 = [
            -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
            p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]
        ]
        A.append(a1)
        A.append(a2)

    A = np.array(A)
    # svd composition
    # the singular value is sorted in descending order
    u, s, v = np.linalg.svd(A)

    # we take the “right singular vector” (a column from V )
    # which corresponds to the smallest singular value
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H

def compute_homography_ransac(C1, C2, M):
    """
    Implements a RANSAC scheme to estimate the homography and the set of inliers.

    Parameters
    ----------
    C1 : numpy array [num_corners x 2]
         corner keypoints for image 1 
    C2 : numpy array [num_corners x 2]
         corner keypoints for image 2
    M  : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 
 
    Returns
    ----------
    H_final : numpy array [3 x 3]
            Homography matrix which maps in point image 1 to image 2 
    M_final : numpy array [num_inlier_matches x 2]
            [cornerIdx1, cornerIdx2] each row contains indices of inlier matches
    """

    max_iter = 100
    min_inlier_ratio = 0.6
    inlier_thres = 5
    P1, P2 = C1[M[:, 0]], C2[M[:, 1]]
    H_final = None
    M_final = None
    for i in range(max_iter):
        rand_four_idx = (np.random.rand(4) * (len(M)-1)).astype(int)
        matches = M[rand_four_idx]
        H = calculate_homography_four_matches(C1[matches[:, 0]],  C2[matches[:, 1]])
        residual = compute_residual(P1, P2, H)
        inliers_idx = np.where(residual < inlier_thres)
        if M_final is None or  len(M_final) < len(inliers_idx):
            M_final = M[inliers_idx]
            H_final = H 

        if len(M_final) / len(residual) >min_inlier_ratio:
            print("Found best homofraphy by exceeding min_inlier_ratio teshold!")
            break

    return H_final, M_final

