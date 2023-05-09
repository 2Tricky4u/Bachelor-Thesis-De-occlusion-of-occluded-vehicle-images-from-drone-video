import cv2
import numpy as np


# Inspired from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def image_fill(image, width, height, color):
    # 255 = blank 0 = black
    bg = np.zeros([width, height], np.uint8)
    bg[:, :] = color
    h1, w1 = image.shape[:2]
    yoff = round((height - h1) / 2)
    xoff = round((width - w1) / 2)
    result = bg.copy()
    patch = image
    if len(image.shape) == 3:
        patch = image[:,:,0]
    elif len(image.shape) == 2:
        patch = image[:,:]
    result[yoff:yoff + h1, xoff:xoff + w1] = patch
    return result
