import cv2
from skimage import data
import matplotlib.pyplot as plt
from noise import noise_gaussian

gas_std = 20;
t_image = data.astronaut()
t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
t_image_ori = t_image
t_image = noise_gaussian(gas_std, t_image)
blur_0 = cv2.blur(t_image,(3,3))
blur_1 = cv2.blur(t_image,(11,11))
blur_2 = cv2.blur(t_image,(21,21))

# b,g,r = cv2.split(blur)
# blur = cv2.merge([r,g,b])
# cv2.imshow('blur',blur)
# cv2.waitKey(0)

plt.figure(figsize=(10,3))
plt.suptitle('Boxfilter')
plt.subplot(1,4,1)
plt.imshow(t_image_ori,cmap='gray')
plt.title('Original')
plt.subplot(1,4,2)
plt.imshow(t_image,cmap='gray')
plt.title(f'Noise = {gas_std}')
plt.subplot(1,4,3)
plt.imshow(blur_1,cmap='gray')
plt.title('3x3 Filter')
plt.subplot(1,4,4)
plt.imshow(blur_2,cmap='gray')
plt.title('21x21 Filter')
plt.tight_layout
plt.show()