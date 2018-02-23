from PIL import Image
import scipy.stats as st
import numpy as np
import math

def load_image(file_name):
    """Load an image into a numpy array."""
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(data, file_name):
    """Saves the data as an image."""
    img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
    img.save(file_name)


def w_average(pix):
    """Weigted grayscale of the image."""
    return 0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2]


def rgb2grey(rgb_img):
    """Creates a grayscale of an RGB image."""
    grey = np.zeros((img.shape[0], img.shape[1]))
    # get row number
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            grey[row][col] = w_average(img[row, col, :])

    return grey


def convolve(data, convf):
    fsize = int((convf.shape[0]) / 2)

    print(data.shape[0])
    out = np.zeros((data.shape[0] - 2 * fsize, data.shape[1] - 2 * fsize))
    
    for row in range(fsize, out.shape[0]+fsize):
        for col in range(fsize, out.shape[1]+fsize):
            out[row-fsize,col-fsize] = np.sum(data[row-fsize:row+fsize + 1, col-fsize:col+fsize + 1] * convf)

    return out

def gkern(kernlen=31, nsig=10, thresh=0.001):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    #kernel = kernel / np.max( kernel.ravel() )

    shape = kernel.shape
    kernel = kernel.ravel()

    inds = np.where( kernel < thresh )[0]
    kernel[ inds ] = 0.0

    kernel = kernel.reshape( shape )

    return kernel

def convolve_magnitude(data, convf, shape_x, shape_y):
    """Applies filters and collects their euclidian magnitude."""

    out = np.zeros((data.shape[0] - shape_x, data.shape[1] - shape_y))

    for row in range(shape_x, out.shape[0]):
        for col in range(shape_y, out.shape[1]):
            """Beware of the super line, maybe this should be split up abit xD."""
            out[row, col] = int(math.sqrt(sum(map(lambda filtr: math.pow(np.sum(data[row:row + shape_x, col: col + shape_y] * filtr), 2), convf))))
                                   
    return out


img = load_image("res/lenna.png")

grey = rgb2grey(img)

vertical_edge_filter = np.array(
    [[1.0, 1.0, 1.0, 1.0, 1.0],
     [2.0, 2.0, 2.0, 2.0, 2.0],
     [0.0, 0.0, 0.0, 0.0, 0.0],
     [-2.0, -2.0, -2.0, -2.0, -2.0],
     [-1.0, -1.0, -1.0, -1.0, -1.0]])

horizontal_edge_filter = np.array(
    [[1.0, 2.0, 0.0, -2.0, -1.0],
     [1.0, 2.0, 0.0, -2.0, -1.0],
     [1.0, 2.0, 0.0, -2.0, -1.0],
     [1.0, 2.0, 0.0, -2.0, -1.0],
     [1.0, 2.0, 0.0, -2.0, -1.0]])

"""
Can maninipulate the convolve_magnitude function for this perticular case to get the
actual angle of edges but taking TAN och the magnitude of the vertical_filter and horizontal_filter!
I might add that later.
"""

sobel_filter = convolve_magnitude(grey, [vertical_edge_filter, horizontal_edge_filter], 5, 5)


simple_filter = np.array([[0,0,.1,0,0],[0, .1, .3, .1, 0], [.1, .3, .5, .3 ,.1], [0, .1, .3, .1, 0], [0,0,.1,0,0]])
simple_filter = simple_filter/np.sum(simple_filter)

blur_filter = np.array(
    [[0.00000067 , 0.00002292 , 0.00019117 , 0.00038771 , 0.00019117 , 0.00002292 , 0.00000067],
     [0.00002292 , 0.00078634 , 0.00655965 , 0.01330373 , 0.00655965 , 0.00078633 , 0.00002292],
     [0.00019117 , 0.00655965 , 0.05472157 , 0.11098164 , 0.05472157 , 0.00655965 , 0.00019117],
     [0.00038771 , 0.01330373 , 0.11098164 , 0.22508352 , 0.11098164 , 0.01330373 , 0.00038771],
     [0.00019117 , 0.00655965 , 0.05472157 , 0.11098164 , 0.05472157 , 0.00655965 , 0.00019117],
     [0.00002292 , 0.00078633 , 0.00655965 , 0.01330373 , 0.00655965 , 0.00078633 , 0.00002292],
     [0.00000067 , 0.00002292 , 0.00019117 , 0.00038771 , 0.00019117 , 0.00002292 , 0.00000067]])

gblur_filter = gkern()

edge_filter = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

blured = convolve(grey, gblur_filter)
edges = convolve(grey, edge_filter)

save_image(sobel_filter, "lenna-sobel.png")
save_image(grey, "lenna.png")

