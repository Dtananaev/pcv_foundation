import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from helpers.corner_detector  import compute_corners


if __name__=="__main__":
    image_path = "checkerboard.png"

    # Load image
    I_checkerboard = np.asarray(Image.open(image_path))
    print(f"image {I_checkerboard.shape}")
    
    # Make it grayscale
    T_harris = 50  
    T_shi_tomasi = 20
    corners_harris = compute_corners(I_checkerboard, 'harris', T_harris)
    corners_shi_tomasi = compute_corners(I_checkerboard, 'shi-tomasi', T_shi_tomasi)

    # Visualize the detections by plotting them over the image
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(I_checkerboard, cmap='gray')
    plt.plot(corners_harris[:,0], corners_harris[:,1], 'rx', markersize=12)
    plt.xlabel("Harris Corners")

    plt.subplot(1,2,2)
    plt.imshow(I_checkerboard, cmap='gray')
    plt.plot(corners_shi_tomasi[:,0], corners_shi_tomasi[:,1], 'rx', markersize=12)
    plt.xlabel("Shi-Tomasi Corners")
    plt.show()