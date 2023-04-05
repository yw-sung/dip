import cv2
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
from noise import noise_gaussian
from noise import noise_sap


def meanfilter():
    ori_img = data.camera()
    src_img = data.camera().astype(float)
    std = 30
    p = 0.05


    gas_img = noise_gaussian(std, ori_img)
    sap_img = noise_sap(p, src_img)

    def contraharmonic_mean(img, ksize, Q):
        img = img.astype(float)
        num = np.power(img, Q+1)
        den = np.power(img, Q)
        kernel = np.full(ksize, 1.0)
        out_img = cv2.filter2D(num, -1, kernel) / cv2.filter2D(den, -1, kernel)
        out_img = np.uint8(np.clip(out_img,0,255))
        return out_img


    # The geometric mean can also be expressed as the exponential of the arithmetic mean of logarithms.
    def gmf(src_img, ksize):
        src_img=src_img.astype(float)
        print(src_img)
        h, w = src_img.shape[:2]
        padsize = int((ksize-1)/2)
        pad_img = cv2.copyMakeBorder(src_img, *[padsize]*4, cv2.BORDER_DEFAULT)
        out_img = np.zeros_like(src_img)
        for r in range(h):
            for c in range(w):
                out_img[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
        out_img = np.uint8(out_img)
        return out_img

    # amf_img = cv2.blur(gas_img, (3, 3))
    amf_img = contraharmonic_mean(gas_img, (3,3), 0.0)
    hmf_img = contraharmonic_mean(gas_img, (3,3), 1.0)
    gmf_img = gmf(gas_img, 3)

    # Plot Gaussian noise processed img
    plt.figure(figsize=(10,5), dpi=300)
    plt.suptitle(f'Mean Filter - Gaussian Noise(std = {std})', fontsize = 20)
    plt.subplot(2,3,1)
    plt.title('Image', fontsize=16)
    plt.imshow(ori_img, cmap='gray')
    plt.subplot(2,3,2)
    plt.title('Gaussian noise', fontsize=16)
    plt.imshow(gas_img, cmap='gray')
    plt.subplot(2,3,3)
    plt.title('Arithmetic mean', fontsize=16)
    plt.imshow(amf_img, cmap='gray')
    plt.subplot(2,3,4)
    plt.title('Geometric mean', fontsize=16)
    plt.imshow(gmf_img, cmap='gray')
    plt.subplot(2,3,5)
    plt.title('Harmonic mean', fontsize=16)
    plt.imshow(hmf_img, cmap='gray')
    plt.tight_layout()
    plt.show()


    # amf_img = cv2.blur(gas_img, (3, 3))
    amf_img = contraharmonic_mean(sap_img, (3,3), 0.0)
    hmf_img = contraharmonic_mean(sap_img, (3,3), -1.0)
    gmf_img = gmf(sap_img, 3)

    # Plot Salt-and-Pepper noise processed img
    plt.figure(figsize=(10,5), dpi=300)
    plt.suptitle(f'Mean Filter - Salt and Pepper Noise(p = {p})', fontsize = 20)
    plt.subplot(2,3,1)
    plt.title('Image', fontsize=16)
    plt.imshow(ori_img, cmap='gray')
    plt.subplot(2,3,2)
    plt.title('Salt-and-Pepper noise', fontsize=16)
    plt.imshow(sap_img, cmap='gray')
    plt.subplot(2,3,3)
    plt.title('Arithmetic mean', fontsize=16)
    plt.imshow(amf_img, cmap='gray')
    plt.subplot(2,3,4)
    plt.title('Geometric mean', fontsize=16)
    plt.imshow(gmf_img, cmap='gray')
    plt.subplot(2,3,5)
    plt.title('Harmonic mean', fontsize=16)
    plt.imshow(hmf_img, cmap='gray')
    plt.tight_layout()
    plt.show()
