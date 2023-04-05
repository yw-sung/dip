import cv2
from skimage import data
import matplotlib.pyplot as plt


def edgedetection():
    src_img = data.brick()
    # src_img = data.gravel()


    # cv2.Sobel(srcimage, ddepth, dx, dy, ksize, scale, delta, bordertpye)
    sobel_dx = cv2.Sobel(src_img, cv2.CV_8U, 1, 0, 3)
    sobel_dy = cv2.Sobel(src_img, cv2.CV_8U, 0, 1, 3)

    # cv2.Laplacian(srcimage, ddepth, ksize, scale, delta, bordertpye)
    laplacian = cv2.Laplacian(src_img, cv2.CV_8U, ksize=3)

    plt.figure(figsize=(7, 2))
    plt.suptitle("Sobel and Laplacian Kernels", fontsize=20)
    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(src_img, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title('Sobel dx')
    plt.imshow(sobel_dx, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title('Sobel dy')
    plt.imshow(sobel_dy, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title('Laplacian')
    plt.imshow(laplacian, cmap='gray')
    plt.tight_layout()
    plt.show()