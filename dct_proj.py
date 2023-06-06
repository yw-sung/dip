import copy

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
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    Y = yuv[:, :, 0]
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    I_max = np.max(np.max(Y))
    print(I_max)

    imsize = Y.shape

    new_dct_Y = np.zeros(imsize)
    dct_Y = np.zeros(imsize)
    dct_U = np.zeros(imsize)
    dct_V = np.zeros(imsize)

    Y2 = np.float32(Y)
    U2 = np.float32(U)
    V2 = np.float32(V)

    tilde_Y = np.ones(imsize).astype(np.float32)
    tilde_U = np.ones(imsize).astype(np.float32)
    tilde_V = np.ones(imsize).astype(np.float32)

    in_dctY = np.zeros(imsize)
    in_dctU = np.zeros(imsize)
    in_dctV = np.zeros(imsize)

    std = 0
    kappa = 0
    for i in np.r_[:imsize[0]:block_size]:
        # std = 0
        for j in np.r_[:imsize[1]:block_size]:
            # processing 8 by 8 block
            local_dct_Y = np.zeros((block_size, block_size))
            local_dct_U = np.zeros((block_size, block_size))
            local_dct_V = np.zeros((block_size, block_size))

            local_dct_Y[0:block_size, 0:block_size] = cv2.dct(Y2[i:(i + block_size), j:(j + block_size)]).copy()
            local_dct_U[0:block_size, 0:block_size] = cv2.dct(U2[i:(i + block_size), j:(j + block_size)]).copy()
            local_dct_V[0:block_size, 0:block_size] = cv2.dct(V2[i:(i + block_size), j:(j + block_size)]).copy()

            local_dct_Y = local_dct_Y * (1/block_size)
            mean = local_dct_Y[0, 0]
            # mean = dct_Y[i, j] / block_size
            # print(f'mean[{i}, {j} = {mean}')

            std = 0
            for k in range(0, block_size):
                for l in range(0, block_size):
                    # print(f'{i+k}, {j+l}')
                    # inner_std = (dct_Y[i + k, j + l] / block_size)**2
                    inner_std = (local_dct_Y[k, l].copy()) ** 2
                    std = std + inner_std
                    # tilde_U[i+k, j+l] = local_dct_U[k, l]
                    # tilde_V[i+k, j+l] = local_dct_V[k, l]
                    if k == (block_size - 1) and l == (block_size - 1):
                        std = np.sqrt(std - (mean) ** 2)
                        # kappa = (tau(dct_Y[i, j] / (block_size * I_max))) / (dct_Y[i, j] / block_size * I_max)
                        kappa = (tau(local_dct_Y[0, 0] / (I_max))) / (local_dct_Y[0, 0] / I_max)
                        arg2 = B_max / (mean + 1 * std)
                        kappa = compare(kappa, arg2, 'min')
                        kappa = compare(kappa, 1, 'max')
                        local_dct_Y = local_dct_Y * block_size
                        local_dct_Y[0, 0] = local_dct_Y[0,0] * kappa
                        # local_dct_U[0, 0] = local_dct_U[0, 0] - (128 * block_size)
                        # local_dct_V[0, 0] = local_dct_V[0, 0] - (128 * block_size)
                        # kappa_matrix[i+(k-7), j+(l-7)] = kappa
                        # print(kappa)
                    # tilde_Y
            dct_Y[i:(i + block_size), j:(j + block_size)] = local_dct_Y[0:block_size, 0:block_size]
            dct_U[i:(i + block_size), j:(j + block_size)] = local_dct_U[0:block_size, 0:block_size]
            dct_V[i:(i + block_size), j:(j + block_size)] = local_dct_V[0:block_size, 0:block_size]
    # tilde_Y = np.multiply(kappa_matrix, dct_Y)
    # tilde_Y = np.multiply(block_size, tilde_Y)
    tilde_Y = np.multiply(tilde_Y, dct_Y)
    tilde_U = np.multiply(tilde_U, dct_U)
    tilde_V = np.multiply(tilde_V, dct_V)

    for i in np.r_[:imsize[0]:block_size]:
        for j in np.r_[:imsize[1]:block_size]:
            in_dctY[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_Y[i:(i + block_size), j:(j + block_size)]).copy()
            in_dctU[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_U[i:(i + block_size), j:(j + block_size)]).copy()
            in_dctV[i:(i + block_size), j:(j + block_size)] = cv2.idct(tilde_V[i:(i + block_size), j:(j + block_size)]).copy()

    in_dct = np.zeros(orisize)
    in_dct[:, :, 0] = in_dctY[:]
    in_dct[:, :, 1] = in_dctU[:]
    in_dct[:, :, 2] = in_dctV[:]

    in_dct = cv2.cvtColor(in_dct.astype(np.uint8), cv2.COLOR_YUV2BGR)
    in_dct = cv2.cvtColor(in_dct.astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.figure()
    # plt.imshow(in_dct)
    plt.imshow(np.hstack((rgb, in_dct)))
    # plt.imshow(kappa_matrix.astype(np.uint8), cmap='gray')
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
    # if x < 0 or x > 1:
    #     print("tau function Input 범위 초과")
    # else:
    return x * (2 - x)


def eta(x, r):
    # if x < 0 or x > 1:
    #     print("eta function Input 범위 초과")
    # else:
    return (x ** (1 / r) + (1 - (1 - x) ** (1 / r))) / 2


def psi(x, m, n, p1, p2):
    # if x < 0 or x > 1 or m > 1 or n > 1 or p1 < 0 or p2 < 0:
    #     print("eta function Input 범위 초과")
    # else:
    if 0 <= x <= m:
        return n * (1 - (1 - (x / m) ** p1))
    elif m <= x <= 1:
        return n + (1 - n) * ((x - m) / (1 - m)) ** p2


def compare(x, y, text):
    if text == 'min':
        if x > y:
            return y
        else:
            return x
    if text == 'max':
        if x > y:
            return x
        else:
            return y


def process(Y2, U2, V2, v_blocksize):
    img_size = Y2.shape
    local_dct_Y = np.zeros((v_blocksize, v_blocksize))
    local_dct_U = np.zeros((v_blocksize, v_blocksize))
    local_dct_V = np.zeros((v_blocksize, v_blocksize))
    for i in np.r_[:img_size[0]:v_blocksize]:
        for j in np.r_[:img_size[1]:v_blocksize]:
            local_dct_Y[0:v_blocksize, 0:v_blocksize] = cv2.dct(Y2[i:(i + v_blocksize), j:(j + v_blocksize)])
            local_dct_U[0:v_blocksize, 0:v_blocksize] = cv2.dct(U2[i:(i + v_blocksize), j:(j + v_blocksize)])
            local_dct_V[0:v_blocksize, 0:v_blocksize] = cv2.dct(V2[i:(i + v_blocksize), j:(j + v_blocksize)])
    return local_dct_Y, local_dct_U, local_dct_V