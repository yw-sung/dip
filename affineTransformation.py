import cv2
from skimage import data
import matplotlib.pyplot as plt
import numpy as np

def affine():
    t_image = data.horse()
    print(t_image.shape)
    t_image = t_image.astype(np.float32)
    # Get Size of the image (328,400, 2ch)
    rows, cols = t_image.shape[:2]

    t_image = cv2.cvtColor(t_image, cv2.COLOR_GRAY2RGB)
    print(t_image.shape)

    pt1 = np.float32([[320,100], [50, 40], [150, 300]])
    pt2 = np.float32([[300,200], [150, 50], [50, 280]])

    # Make B, G ,R circle at each point
    cv2.circle(t_image, (320,100), 10, (255,0,0), -1)
    cv2.circle(t_image, (50,40), 10, (0,255,0), -1)
    cv2.circle(t_image, (150,300), 10, (0,0,255), -1)

    M = cv2.getAffineTransform(pt1, pt2)
    print(M)
    tt_image = cv2.warpAffine(t_image, M, (cols, rows))

    # cv2.imshow('Original', t_image)
    # cv2.imshow('Affine', tt_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    b, g, r = cv2.split(t_image)
    t_image = cv2.merge([r,g,b])
    b, g, r = cv2.split(tt_image)
    tt_image = cv2.merge([r,g,b])

    plt.subplot(1,2,1)
    plt.imshow(t_image)
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(tt_image)
    plt.title('Affine')
    plt.tight_layout()
    plt.show()