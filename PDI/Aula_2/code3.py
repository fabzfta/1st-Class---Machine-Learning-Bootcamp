
import matplotlib.pyplot as plt

def load_building_image():
    return plt.imread("imgs//toa-sharp-def-3.jpg")

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

building_image = load_building_image()

# Import Gaussian filter
from skimage.filters import gaussian

# Apply filter
gaussian_image = gaussian(building_image, sigma=1, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")

gaussian_image2 = gaussian(building_image, sigma=5, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image2, "Reduced sharpness Gaussian")

gaussian_image3 = gaussian(building_image, sigma=10, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image3, "Reduced sharpness Gaussian")

