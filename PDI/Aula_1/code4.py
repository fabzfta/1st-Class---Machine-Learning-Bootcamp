# from PDI.src.pdi_utils import show_image , load_chess_image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_chess_image():
    return plt.imread("imgs//chess_gray.png")


# Import the otsu threshold function

chess_pieces_image = load_chess_image()

# Make the image grayscale using rgb2gray
chess_pieces_image_gray = rgb2gray(chess_pieces_image)

#show original image
show_image(chess_pieces_image,'Original image')

# Obtain the optimal threshold value with otsu
thresh = threshold_otsu(chess_pieces_image_gray)

# Apply thresholding to the image
binary = chess_pieces_image_gray > thresh

# Show the binary image
show_image(binary, 'Binary image')

plt.imsave('test.jpg', binary)