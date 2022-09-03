# from PDI.src.pdi_utils import load_lena
import matplotlib.pyplot as plt

# from PDI.src.pdi_utils import load_red_roses,show_image
import matplotlib.pyplot as plt

def load_lena():
    return plt.imread("imgs//lena.png")

def load_red_roses():
    return plt.imread("imgs//red_roses.png")

def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

image = load_red_roses()

# Show original image
show_image(image,'image RGB')

# Obtain the red channel
red_channel = image[:, : , 2]

# Show original image
show_image(red_channel,'image red channel')

# Plot the red histogram with bins in a range of 256
plt.hist(red_channel, bins=30)

# Set title and show
plt.title('Red Histogram')
plt.show()