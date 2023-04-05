import cv2
import numpy as np
from random import random


def noise_gaussian(std, gray_img):
    gray_img = gray_img.astype(float)
    h, w = gray_img.shape
    gaussian = np.random.normal(0, std, (h,w))
    noise = np.zeros((h, w), dtype=np.float64)
    cv2.imshow('noise', gaussian)
    noise = gray_img + gaussian
    noise = np.uint8(np.clip(noise, 0, 255))
    print(noise)
    return noise


def noise_sap(p, img):
    h, w = img.shape
    th = 1 - p
    noise = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            rdn = random()
            if rdn < p:
                noise[i][j] = 0
            elif rdn > th:
                noise[i][j] = 255
            else:
                noise[i][j] = img[i][j]
    noise = np.uint8(np.clip(noise, 0, 255))
    return noise
