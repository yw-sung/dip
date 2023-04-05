import cv2
from skimage import data
import matplotlib.pyplot as plt
from noise import noise_gaussian
from noise import noise_sap


def orderfilter():
    std = 30
    p = 0.05
    src_img = data.camera()
    gas_img = noise_gaussian(std, src_img)
    sap_img = noise_sap(p, src_img)

    med_img = cv2.medianBlur(gas_img, 3)
    med_img2 = cv2.medianBlur(sap_img, 3)

    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    min_img = cv2.erode(gas_img, kernel)
    min_img2 = cv2.erode(sap_img, kernel)
    max_img = cv2.dilate(gas_img, kernel)
    max_img2 = cv2.dilate(sap_img, kernel)

    mid_img = 0.5 * (min_img + max_img)
    mid_img2 = 0.5 * (min_img2 + max_img2)

    plt.figure(figsize=(10, 5), dpi=300)
    plt.suptitle(f'Order Filter - Gaussian Noise(std = {std})', fontsize=20)
    plt.subplot(2, 3, 1)
    plt.title('Image', fontsize=16)
    plt.imshow(src_img, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title('Gaussian noise', fontsize=16)
    plt.imshow(gas_img, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title('Median', fontsize=16)
    plt.imshow(med_img, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title('Min', fontsize=16)
    plt.imshow(min_img, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title('Max', fontsize=16)
    plt.imshow(max_img, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title('Midpoint', fontsize=16)
    plt.imshow(mid_img, cmap='gray')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5), dpi=300)
    plt.suptitle(f'Order Filter - Salt-and-Pepper Noise(p = {p})', fontsize=20)
    plt.subplot(2, 3, 1)
    plt.title('Image', fontsize=16)
    plt.imshow(src_img, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.title('Salt-and-Pepper noise', fontsize=16)
    plt.imshow(sap_img, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.title('Median', fontsize=16)
    plt.imshow(med_img2, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.title('Min', fontsize=16)
    plt.imshow(min_img2, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title('Max', fontsize=16)
    plt.imshow(max_img2, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.title('Midpoint', fontsize=16)
    plt.imshow(mid_img2, cmap='gray')
    plt.tight_layout()
    plt.show()