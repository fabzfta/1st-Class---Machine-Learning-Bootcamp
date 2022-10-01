# Import the required module
import matplotlib.pyplot as plt
from skimage import exposure

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_chest_ray_x():
    return plt.imread("imgs//contrast_00000109_005.png")

chest_xray_image = load_chest_ray_x()
# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=255)
plt.show()

# Use histogram equalization to improve the contrast
xray_image_eq =  exposure.equalize_hist(chest_xray_image)


# Show the resulting image
show_image(xray_image_eq, 'Resulting image')

plt.title('Histogram of image - Euqlized')
plt.hist(xray_image_eq.ravel(), bins=255)
plt.show()