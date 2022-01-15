from matplotlib.pyplot import jet
import numpy as np


def compute_descriptors(I, corners):
    """
    Computes a 128 bit SIFT like descriptor.
    Parameters
    ----------
    I : float [MxN]
        grayscale image as a 2D numpy array
    corners : numpy array [num_corners x 2] 
              Coordinates of the detected corners. 
    
    Returns
    -------
    D : numpy array [num_corners x 128]
        128 bit descriptors  corresponding to each corner keypoint

    """
    descriptors = []
    height, width = np.asarray(I).shape
    #print(f"height {height}, width {width}")
    patch_size = 16
    half_patch = int(patch_size/2)
    boundary_image = np.zeros((height + patch_size + 1, width + patch_size + 1), dtype=float)
    boundary_image[half_patch:height + half_patch, half_patch:width + half_patch] = I

    orientations = np.asarray([[1, 0], [0.5, 0.5], [0, 1], [-0.5, 0.5], [-1, 0], [-0.5, -0.5], [0, -1], [0.5, -0.5]], dtype=float)
    for corner in corners:
        corner += half_patch
        c_min = corner - half_patch
        c_max = corner +  half_patch    
        # Forward difference
        grad_x = boundary_image[c_min[0]:c_max[0], c_min[1]:c_max[1]] - boundary_image[c_min[0]:c_max[0], c_min[1]+1:c_max[1]+1]
        grad_y = boundary_image[c_min[0]:c_max[0], c_min[1]:c_max[1]] - boundary_image[c_min[0]+1:c_max[0]+1, c_min[1]:c_max[1]]
        grad = np.concatenate((grad_x[..., None], grad_y[..., None]), axis=-1)
        squares = []
        # Get squares [num_corners, 16, 16, 2] -> [num_corners, 16, 4, 4, 2]
        for i in range(0, patch_size, 4):
            for j in range(0, patch_size, 4):
                squares.append(grad[i:i + 4, j:j + 4, :])
        squares = np.asarray(squares)
        sq_shape = squares.shape
        squares = np.reshape(squares, (sq_shape[0], -1, 2))
        magnitude = np.linalg.norm(squares, axis=-1) / 255.0
        hist = np.inner(orientations, squares)
        hist = np.argmax(hist, axis=0)
        res_hist = np.zeros( (patch_size, 8))
        for i in range(patch_size):
            for j in range(patch_size):
                idx = int(hist[i, j])
                res_hist[i, idx] += magnitude[i, j]
        descriptors.append(np.reshape(res_hist, (-1)))
    return np.asarray(descriptors)



def compute_matches(D1, D2, T = 1.0):
    """
    Computes matches for two images using the descriptors.
    Uses the Lowe's criterea to determine the best match.

    Parameters
    ----------
    D1 : numpy array [num_corners x 128]
         descriptors for image 1 corners
    D2 : numpy array [num_corners x 128]
         descriptors for image 2 corners
 
    Returns
    ----------
    M : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 
    """
    M = []
    dist = np.linalg.norm(D1[:, None, :] - D2[None, ...], axis=-1)

    for id1, d in enumerate(dist):
        idx = np.argpartition(d, 2, axis=-1)
        q_1 = d[idx[0]]
        q_2 = d[idx[1]]

        if q_1/q_2 > 0.5 or q_1 > T:
            continue
        M.append([id1, idx[0]])


    return np.asarray(M)
