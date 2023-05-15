import cv2
import numpy as np

# Inspired from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
"""Resize an image while keeping its ratio to not deform it

:param image: the image to resize
:param width: the wanted width
:param height: the wanted height
:param inter: default= cv2.INTER_AREA

:returns: the image resized
"""


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
    elif height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        rh = height / float(h)
        rw = width / float(w)
        ratio = min(rh, rw)
        dim = (int(w * ratio), int(h * ratio))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


"""Fill an image of the color given with the also given dimension

:param image: the image to fill
:param width: the wanted width
:param height: the wanted height
:param color: the color can be from blank to black (grayscale)

:returns: the image filled with the new dimension
"""


def image_fill(image, width, height, color):
    # 255 = blank 0 = black
    bg = np.zeros([width, height, 3], np.uint8)
    bg[:, :] = color
    h1, w1 = image.shape[:2]
    yoff = round((height - h1) / 2)
    xoff = round((width - w1) / 2)
    result = bg.copy()
    patch = image
    try:
        result[yoff:yoff + h1, xoff:xoff + w1] = patch
    except Exception as err:
        print('Handling run-time error on resize:', err)
        print("----------------")
        print(image.shape)
        print(image.shape[:2])
        print("--------------------")
        print(patch.shape)
        print(result[yoff:yoff + h1, xoff:xoff + w1].shape)
    return result

