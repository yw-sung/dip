import cv2
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
from noise import noise_gaussian
from noise import noise_sap

ori_img = data.camera()
h, w = ori_img.shape[:2]
src_img = data.camera().astype(float)
std = 5
p = 0.2


gas_img = noise_gaussian(std, ori_img)
sap_img = noise_sap(p, src_img)

def contraharmonic_mean(img, ksize, Q):
    num = np.power(img, Q+1)
    den = np.power(img, Q)
    kernel = np.full(ksize, 1.0)
    out_img = cv2.filter2D(num, -1, kernel) / cv2.filter2D(den, -1, kernel)
    out_img = np.uint8(np.rint(out_img))
    return out_img


# amf_img = cv2.blur(gas_img, (3, 3))
amf_img = contraharmonic_mean(gas_img, (3,3), 0.0)
hmf_img = contraharmonic_mean(gas_img, (3,3), -1.0)

# The geometric mean can also be expressed as the exponential of the arithmetic mean of logarithms.
a= np.uint8(np.log(np.clip(gas_img,1,255)))
print(a)
b = cv2.blur(a, (3,3))
print(b)
gmf_img = np.uint8(np.exp(cv2.boxFilter(np.uint8(np.log(gas_img)), -1, (3, 3))))
print(gmf_img)

# Plot Gaussian noise processed img
plt.figure(figsize=(25,10))
plt.subplot(1,5,1)
plt.title('Image')
plt.imshow(ori_img, cmap='gray')
cv2.imshow('1',ori_img)
plt.subplot(1,5,2)
plt.title('Gaussian noise')
plt.imshow(gas_img, cmap='gray')
cv2.imshow('2',gas_img)
plt.subplot(1,5,3)
plt.title('Arithmetic mean')
plt.imshow(amf_img, cmap='gray')
cv2.imshow('3',amf_img)
plt.subplot(1,5,4)
plt.title('Geometric mean')
plt.imshow(gmf_img, cmap='gray')
cv2.imshow('4',gmf_img)
plt.subplot(1,5,5)
plt.title('Harmonic mean')
plt.imshow(hmf_img, cmap='gray')
cv2.imshow('5',hmf_img)
plt.tight_layout()
plt.show()
cv2.waitKey(15)


