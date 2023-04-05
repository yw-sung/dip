import cv2
from skimage import data
import matplotlib.pyplot as plt

def interpolation():
    t_image = data.astronaut()
    t_shape = t_image.shape[:2]

    # Check Size of the image (512x512 3ch)

    tt_image = cv2.resize(t_image, dsize=(150, 150))

    zth_image = cv2.resize(tt_image, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
    bilinear_image = cv2.resize(tt_image, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
    bicubic_image = cv2.resize(tt_image, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)

    images = [t_image, zth_image, bilinear_image, bicubic_image]
    titles = ["Original", "Zth-Order", "Bilinear", "Bicubic"]

    plt.figure(figsize=(10, 3), dpi=300)
    plt.suptitle('Interpolation 150x150 to 500x500', fontsize=20)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image)
        plt.xlim(0, 500)
        plt.ylim(500, 0)
        plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


    def crop(image, left_top, x=50, y=100):
        return image[left_top[0]:(left_top[0] + x), left_top[1]:(left_top[1] + y), :]


    left_tops = [(220, 200)] * 4 + [(90, 120)] * 4 + [(30, 200)] * 4
    plt.figure(figsize=(10, 6), dpi=300)
    plt.suptitle('Cropped', fontsize=20)
    for i, (image, left_top, title) in enumerate(zip(images * 4, left_tops, titles * 4)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(crop(image, left_top))
        if i == 0: # Original
            plt.title(title, fontsize=16)
        if i == 1: # Zth
            plt.title(title, fontsize=16)
        if i == 2: # Bilinear
            plt.title(title, fontsize=16)
        if i == 3: # Bicubic
            plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
