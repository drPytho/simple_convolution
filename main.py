from PIL import Image
import numpy as np


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


img = load_image("res/lenna.png")

grey = rgb2grey(img)

simple_filter = np.array([[0,0,.1,0,0],[0, .1, .3, .1, 0], [.1, .3, .5, .3 ,.1], [0, .1, .3, .1, 0], [0,0,.1,0,0]])
simple_filter = simple_filter/np.sum(simple_filter)
print(simple_filter)

blur_filter = np.array(
    [[0.00000067 , 0.00002292 , 0.00019117 , 0.00038771 , 0.00019117 , 0.00002292 , 0.00000067],
     [0.00002292 , 0.00078634 , 0.00655965 , 0.01330373 , 0.00655965 , 0.00078633 , 0.00002292],
     [0.00019117 , 0.00655965 , 0.05472157 , 0.11098164 , 0.05472157 , 0.00655965 , 0.00019117],
     [0.00038771 , 0.01330373 , 0.11098164 , 0.22508352 , 0.11098164 , 0.01330373 , 0.00038771],
     [0.00019117 , 0.00655965 , 0.05472157 , 0.11098164 , 0.05472157 , 0.00655965 , 0.00019117],
     [0.00002292 , 0.00078633 , 0.00655965 , 0.01330373 , 0.00655965 , 0.00078633 , 0.00002292],
     [0.00000067 , 0.00002292 , 0.00019117 , 0.00038771 , 0.00019117 , 0.00002292 , 0.00000067]])

edge_filter = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

blured = convolve(grey, blur_filter)
edges = convolve(grey, edge_filter)

save_image(grey, "lenna.png")
save_image(blured, "lenna-blured.png")
save_image(edges, "lenna-edges.png")

