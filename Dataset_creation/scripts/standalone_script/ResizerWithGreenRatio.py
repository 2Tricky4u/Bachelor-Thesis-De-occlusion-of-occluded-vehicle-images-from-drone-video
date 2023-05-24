import math
import os
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
#%%


#https://stackoverflow.com/questions/66757199/color-percentage-in-image-for-python-using-opencv
def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def ratio_of_green(img):
    green = [130, 170, 50]
    diff = 60
    boundaries = [([green[2], green[1] - diff, green[0] - diff],
                   [green[2] + diff, green[1] + diff, green[0] + diff])]
    for (lower, upper) in boundaries:

        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(img, lower, upper)

    return cv2.countNonZero(mask)/(img.size/3)

def main():
    original_path = "./v_patches/"
    files = get_files_from_folder(original_path)

    path = "./resized/"
    # creates dir
    if not os.path.exists(path):
        os.makedirs(path)

    f = open('file.txt', 'w')
    for file in files:
        act_path = os.path.join(original_path, file)
        next_path = os.path.join(path, file)
        img = cv2.imread(act_path)
        shape = img.shape
        line = str(shape)
        f.write(line + '\n')
        x_center = math.floor(shape[0] / 2)
        y_center = math.floor(shape[1] /2 )
        offsetx = math.floor(x_center / 2)
        offsety = math.floor(y_center / 2)
        x_start = x_center - offsetx
        x_end = x_center + offsetx
        y_start = y_center - offsety
        y_end = y_center + offsety
        center = img[x_start : x_end, y_start:y_end, :]
        img_resized = image_resize(img, 128, 128)
        translated_img = image_fill(img_resized, 128, 128, [0, 0, 0])
        if ratio_of_green(img) < 0.05 and ratio_of_green(center) < 0.01:
            cv2.imwrite(next_path, translated_img)
        else:
            print(ratio_of_green(img))
            print(ratio_of_green(center))
            cv2.imwrite(os.path.join("./green/", file), translated_img)
    f.close()

if __name__ == '__main__':
    print(" - Start the creation of the dataset with the following parameters: - ")
    main()
