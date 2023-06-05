import cv2
from skimage import data
import numpy as np
import matplotlib.pyplot as plt


def dct_proj():
    block_size = 8;
    sigma_thresh = 15;
    B_max = 255;

    mod_h = 0;
    mod_w = 0;
    # original = data.astronaut()
    original = cv2.imread('sample/image16.jpg')

    orisize = original.shape

    # orisize[0] is height, and orisize[1] is width
    want, nay = divmod(orisize[0], block_size)
    if nay != 0:
        mod_h = block_size * (want + 1)
    else:
        mod_h = orisize[0]
    want, nay = divmod(orisize[1], block_size)
    if nay != 0:
        mod_w = block_size * (want + 1)
    else:
        mod_w = orisize[1]

    original = cv2.resize(original, dsize=(mod_w, mod_h))
    orisize = original.shape
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(rgb)
    plt.show()
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    Y = yuv[:, :, 0]
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    I_max = np.max(Y)
    print(I_max)

    imsize = Y.shape

    dct_Y = np.zeros(imsize)
    dct_U = np.zeros(imsize)
    dct_V = np.zeros(imsize)

    Y2 = np.float32(Y)
    U2 = np.float32(U)
    V2 = np.float32(V)
    tilde_Y = np.zeros(imsize)
    tilde_U = np.zeros(imsize)
    tilde_V = np.zeros(imsize)
    kappa_matrix = np.zeros(imsize)

    for i in np.r_[:imsize[0]:block_size]:
        # mean = 0
        std = 0
        for j in np.r_[:imsize[1]:block_size]:
            dct_Y[i:(i + block_size), j:(j + block_size)] = cv2.dct(Y2[i:(i + block_size), j:(j + block_size)])
            dct_U[i:(i + block_size), j:(j + block_size)] = cv2.dct(U2[i:(i + block_size), j:(j + block_size)])
            dct_V[i:(i + block_size), j:(j + block_size)] = cv2.dct(V2[i:(i + block_size), j:(j + block_size)])
            mean = dct_Y[i, j] / block_size
            tilde_U[i, j] = dct_U[i, j] - 128 * block_size
            tilde_V[i, j] = dct_V[i, j] - 128 * block_size

            for k in range(0, block_size):
                for l in range(0, block_size):
                    # print(f'{i+k}, {j+l}')
                    inner_std = (dct_Y[i + k, j + l] / block_size) ** 2
                    std = std + inner_std
                    tilde_U[i + k, j + l] = dct_U[i + k, j + l]
                    tilde_V[i + k, j + l] = dct_V[i + k, j + l]
            # if dct[i,j]>sigma_thresh
            #     for k in np.r_[:imsize[0]:block_size/2]:
            #         for l in np.r_[:imsize[1]:block_size/2]:
            #             dct[i:(i + block_size/2), j:(j + block_size)] = cv2.dct(
            #                 Y2[i:(i + block_size), j:(j + block_size)])
            std = np.sqrt(std - mean ** 2)
            pre_kappa = tau(dct_Y[i, j] / (block_size * I_max))
            if pre_kappa == 0:
                kappa = 0
            else:
                kappa = np.float32(tau(dct_Y[i, j] / (block_size * I_max))) / (dct_Y[i, j] / block_size * I_max)
            arg2 = B_max / (mean + kappa * std)
            if kappa > arg2:
                kappa = arg2
            if kappa < 1:
                kappa = 1
            for b1 in range(0, block_size):
                for b2 in range(0, block_size):
                    kappa_matrix[i + b1, j + b2] = kappa
    # print(kappa_matrix)
    tilde_Y = np.multiply(kappa_matrix, dct_Y)
    # tilde_Y = dct_Y

    in_dctY = np.zeros(imsize)
    in_dctU = np.zeros(imsize)
    in_dctV = np.zeros(imsize)

    for i in np.r_[:imsize[0]:block_size]:
        for j in np.r_[:imsize[1]:block_size]:
            in_dctY[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_Y[i:(i + block_size), j:(j + block_size)])
            in_dctU[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_U[i:(i + block_size), j:(j + block_size)])
            in_dctV[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_V[i:(i + block_size), j:(j + block_size)])

    in_dct = np.zeros(orisize)
    in_dct[:, :, 0] = in_dctY.astype(np.uint8)
    in_dct[:, :, 1] = in_dctU.astype(np.uint8)
    in_dct[:, :, 2] = in_dctV.astype(np.uint8)

    in_dct = cv2.cvtColor(in_dct.astype(np.uint8), cv2.COLOR_YUV2BGR)
    in_dct = cv2.cvtColor(in_dct, cv2.COLOR_BGR2RGB)
    plt.figure()
    # plt.imshow(in_dct)
    plt.imshow(np.hstack((rgb, in_dct)))
    # plt.imshow(kappa_matrix)
    # plt.imshow(np.hstack((original, in_dct)), cmap='gray')
    plt.show()

    show_pos = 64

    # plt.figure()
    # plt.imshow(Y[show_pos:show_pos + block_size, show_pos:show_pos + block_size], cmap='gray')
    # plt.title(f'{block_size}X{block_size} Original Image')
    #
    # plt.figure()
    # plt.imshow(dct[show_pos:show_pos + block_size, show_pos:show_pos + block_size], cmap='gray')
    # plt.title(f'{block_size}X{block_size} DCT Image')

    # plt.show()


def tau(x):
    if x < 0 or x > 1:
        print("tau function Input 범위 초과")
    else:
        return x * (2 - x)


def eta(x, r):
    if x < 0 or x > 1:
        print("eta function Input 범위 초과")
    else:
        return (x ^ (1 / r) + (1 - (1 - x) ^ (1 / r))) / 2


def psi(x, m, n, p1, p2):
    if x < 0 or x > 1 or m > 1 or n > 1 or p1 < 0 or p2 < 0:
        print("eta function Input 범위 초과")
    else:
        if 0 <= x <= m:
            return n * (1 - (1 - (x / m) ^ p1))
        elif m <= x <= 1:
            return n + (1 - n) * ((x - m) / (1 - m)) ^ p2
