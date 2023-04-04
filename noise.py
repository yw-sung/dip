import cv2
import numpy as np


def noise_gaussian(std, gray_img):
    h, w = gray_img.shape
    noise = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            noise_temp = np.random.normal()
            noise_weighted = std * noise_temp
            noise[i][j] = gray_img[i][j] + noise_weighted
    return noise

def noise_sap(std, gray_img):
    h, w = gray_img.shape
    noise = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            noise_temp = np.random.normal()
            noise_weighted = std * noise_temp
            noise[i][j] = gray_img[i][j] + noise_weighted
    return noise