import cv2
from skimage import data
import matplotlib.pyplot as plt
from noise import noise_gaussian, noise_sap

def boxfilter():
    gas_std = 30
    sap_p = 0.2

    t_image = data.astronaut()
    t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
    t_image_gas = noise_gaussian(gas_std, t_image)
    t_image_sap = noise_sap(0.2, t_image)

    blur_ori = cv2.blur(t_image,(3,3))
    blur_gas = cv2.blur(t_image_gas,(3,3))
    blur_sap = cv2.blur(t_image_sap,(3,3))

    plt.figure(figsize=(10,3.5))
    plt.suptitle('3x3 Boxfilter Processed', fontsize=20)
    plt.subplot(1,3,1)
    plt.imshow(blur_ori,cmap='gray')
    plt.title('Non-processed')
    plt.subplot(1,3,2)
    plt.imshow(blur_gas, cmap='gray')
    plt.title(f'Gaussian std = {gas_std}')
    plt.subplot(1,3,3)
    plt.imshow(blur_sap,cmap='gray')
    plt.title(f'Salt-and-pepper p = {sap_p}')
    plt.tight_layout()
    plt.show()