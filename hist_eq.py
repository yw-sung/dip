import cv2
import numpy as np
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def hist_eq():
    t_image = data.coffee()
    # print(t_image.shape)
    # t_image_bin = cv2.cvtColor(t_image, cv2.COLOR_RGB2GRAY)
    t_image_bin = data.camera()
    print(t_image_bin.shape)
    channels = cv2.split(t_image)
    colors = ['b', 'g', 'r']
    hist_bin = cv2.calcHist([t_image_bin], [0], None, [256], [0,256])

    plt.figure(figsize=(10,6), dpi=300)
    plt.suptitle('Histograms', fontsize=20)
    plt.subplot(2,2,1)
    plt.imshow(t_image_bin,cmap=cm.gray)
    plt.title('Grayscale IMG', fontsize=16)
    plt.subplot(2,2,2)
    plt.plot(hist_bin)
    plt.title('Grayscale Hist', fontsize=16)

    eq_bin = cv2.equalizeHist(t_image_bin)
    hist_eq_bin = cv2.calcHist([eq_bin], [0], None, [256], [0, 256])
    plt.subplot(2,2,3)
    plt.imshow(eq_bin, cmap=cm.gray)
    plt.title('Equalized IMG', fontsize=16)
    plt.subplot(2,2,4)
    plt.plot(hist_eq_bin)
    plt.title('Equalized Hist', fontsize=16)
    plt.xlim([0,256])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,3), dpi=300)
    plt.subplot(1,2,1)
    plt.imshow(t_image)
    plt.title('Colored IMG')
    # plt.subplot(2,2,2)
    # plt.title('Colored Hist')
    # for ch, color in zip(channels, colors):
    #     hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    #     plt.plot(hist, color=color)
    eq_b = cv2.equalizeHist(channels[0])
    eq_g = cv2.equalizeHist(channels[1])
    eq_r = cv2.equalizeHist(channels[2])
    eq = cv2.merge([eq_r,eq_g,eq_b])
    # hist_eq_b = cv2.calcHist(eq_b, [0], None, [256], [0, 256])
    # hist_eq_g = cv2.calcHist(eq_g, [0], None, [256], [0, 256])
    # hist_eq_r = cv2.calcHist(eq_r, [0], None, [256], [0, 256])
    plt.subplot(1,2,2)
    plt.imshow(eq)
    plt.title('Equalized IMG')
    # plt.subplot(2,1,4)
    # plt.plot(hist_eq_r)
    # plt.plot(hist_eq_g)
    # plt.plot(hist_eq_b)
    # plt.title('Equalized Hist')
    # plt.xlim([0,256])
    plt.tight_layout()
    plt.show()