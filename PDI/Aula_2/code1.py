# Import threshold and gray convertor functions
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def load_lools_image():
    return plt.imread("imgs//shapes52.jpg")

    
def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

tools_image = load_lools_image()


# Turn the image grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal thresh
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh


# Show the original image
show_image(tools_image, 'original image')
# Show the resulting binary image
show_image(binary_image, 'Binarized image')