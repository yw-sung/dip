import cv2
from skimage import data
import matplotlib.pyplot as plt


def unsharpmask():
    src_img = data.gravel()

    blr_img = cv2.blur(src_img, (3, 3))
    msk_img = src_img - blr_img

    def highboostfilter(src, msk, k):
        out_img = src + k * msk
        return out_img

    spd_img = highboostfilter(src_img, msk_img, 3)

    plt.figure(figsize=(10, 4))
    plt.suptitle('Unsharp Mask and Sharpened Image', fontsize=20)
    plt.subplot(1, 4, 1)
    plt.title('Original')
    plt.imshow(src_img, cmap='gray')
    plt.subplot(1, 4, 2)
    plt.title('Blurred')
    plt.imshow(blr_img, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title('Unsharp Mask')
    plt.imshow(msk_img, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title('Sharpened')
    plt.imshow(spd_img, cmap='gray')
    plt.tight_layout()
    plt.show()
