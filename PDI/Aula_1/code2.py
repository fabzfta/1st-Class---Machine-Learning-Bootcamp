import numpy as np
# from PDI.src.pdi_utils import load_flipped_seville, show_image
import matplotlib.pyplot as plt


def load_flipped_seville():
    return plt.imread("imgs/flipped_seville.png")

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


flipped_seville = load_flipped_seville()

# Show original image
show_image(flipped_seville, 'Seville Flipped')

# Flip the image vertically
seville_vertical_flip = np.flipud(flipped_seville)

# Show image flippped vertically
show_image(seville_vertical_flip, 'Seville Vertical Flipped')

# Flip the image horizontally
seville_horizontal_flip = np.fliplr(flipped_seville)

# Show image flipped horizontally
show_image(flipped_seville, 'Seville horizontal Flipped')

