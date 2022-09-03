import matplotlib.pyplot as plt
# from PDI.src.pdi_utils import load_lena
# Import the try all function
from skimage.filters  import try_all_threshold
# Import the rgb to gray convertor function
from skimage.color import rgb2gray

def load_lena():
    return plt.imread("imgs//lena.png")

lena = load_lena()

# Turn the Lena image to grayscale
grayscale = rgb2gray(lena)

# Use the try all method on the resulting grayscale image
fig, ax = try_all_threshold(grayscale, verbose=False)

# Show the resulting plots
plt.show()