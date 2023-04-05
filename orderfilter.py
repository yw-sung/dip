import cv2
from skimage import data
import matplotlib.pyplot as plt
from noise import noise_gaussian
from noise import noise_sap

std = 10
src_img = data.camera()
gas_img = noise_gaussian(std, src_img)
sap_img = noise_sap(0.05, src_img)

med_img = cv2.medianBlur(gas_img, 3)
med_img_2 = cv2.medianBlur(sap_img, 3)

size = (3, 3)
shape = cv2.MORPH_RECT
kernel = cv2.getStructuringElement(shape, size)

min_img = cv2.erode(gas_img, kernel)
min_img2 = cv2.erode(sap_img, kernel)
max_img = cv2.dilate(gas_img, kernel)
max_img2 = cv2.dilate(sap_img, kernel)

mid_img = 0.5 * (min_img + max_img)
mid_img2 = 0.5 * (min_img2 + max_img2)