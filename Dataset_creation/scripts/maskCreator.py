import math
import random

import shapeCreator
import numpy as np
import cv2
import os
import shapeCreator as sc

from splitter import get_files_from_folder
from utils import randomShape, Shape
from resizer import image_resize


class Properties:
    def __init__(self, FLAGS, dim):
        self.size = sc.size_random(dim, FLAGS.hidden_ratio)
        shape = randomShape(FLAGS.rec, FLAGS.poly, FLAGS.star, FLAGS.circle, FLAGS.ellipse, FLAGS.random)
        self.shape = shape.getShape()
        self.rota = random.randint(0, 359)
        self.rect_ratio = [random.randint(1, 5), random.randint(1, 5)]
        (x, y) = dim
        self.trX = x / 2
        self.trY = y / 2
        self.ellipse_ratio = [random.randint(1, 5), random.randint(1, 5)]
        self.vertices = random.randint(3, 9)
        self.branch = random.randint(3, 9)
        self.scale = [1, 1]
        self.mask_color = FLAGS.maskcol
        r = FLAGS.hidden_ratio
        if self.shape == Shape.RECTANGLE:
            self.size = sc.size_rec(dim, self.rect_ratio, r)

        if self.shape == Shape.POLYGONE:
            self.size = sc.size_poly(dim, r)

        if self.shape == Shape.STAR:
            self.size = sc.size_star(dim, r)

        if self.shape == Shape.CIRCLE:
            self.size = sc.size_circle(dim, r)

        if self.shape == Shape.ELLIPSIS:
            self.size = sc.size_ellipse(dim, self.ellipse_ratio, r)

        if self.shape == Shape.RANDOM:
            self.size = sc.size_random(dim, r)
        (x, y) = define_position(dim, int(self.size))
        self.trX = x
        self.trY = y
        self.size2 = self.size/2


def create_masks(FLAGS):
    train_path = FLAGS.train_data_output[0]
    test_path = FLAGS.test_data_output[0]
    act_train_path = os.path.join(train_path, "gt/")
    act_test_path = os.path.join(test_path, "gt/")
    mask_train_path = os.path.join(train_path, "mask/")
    mask_test_path = os.path.join(test_path, "mask/")

    if not os.path.exists(mask_train_path):
        os.makedirs(mask_train_path)
    if not os.path.exists(mask_test_path):
        os.makedirs(mask_test_path)

    train_files = get_files_from_folder(act_train_path)
    test_files = get_files_from_folder(act_test_path)
    counter = len(train_files)
    znb = math.ceil(math.log10(counter))

    for count, img in enumerate(train_files):
        src = os.path.join(act_train_path, train_files[count])
        dst = os.path.join(mask_train_path, str(count).zfill(znb) + ".jpg")
        image = cv2.imread(src, 0)
        (oh, ow) = image.shape
        p = Properties(FLAGS, image.shape)
        create_mask(oh, ow, p, FLAGS, dst)

    for count, img in enumerate(test_files):
        src = os.path.join(act_test_path, test_files[count])
        dst = os.path.join(mask_test_path, str(count).zfill(znb) + ".jpg")
        image = cv2.imread(src, 0)
        (oh, ow) = image.shape
        p = Properties(FLAGS, image.shape)
        create_mask(oh, ow, p, FLAGS, dst)


def define_position(dim, size):
    (h, w) = dim
    delta = int(size / 2)
    w1 = max(delta + 1, int(w) - delta)
    h1 = max(delta + 1, int(h) - delta)
    x = random.randint(delta, w1)
    y = random.randint(delta, h1)
    return x, y


def create_mask(w, h, p, FLAGS, dst):
    img, ctx = shapeCreator.makeCanvas((h, w), FLAGS.bgcol)
    ctx.save()
    ctr = [0, 0]
    if p.shape == Shape.RECTANGLE:
        ctx = sc.rectangle(ctx, ctr, p)
    elif p.shape == Shape.POLYGONE:
        ctx = sc.poly(ctx, ctr, p)
    elif p.shape == Shape.STAR:
        ctx = sc.star(ctx, ctr, p)
    elif p.shape == Shape.CIRCLE:
        ctx = sc.circle(ctx, ctr, p)
    elif p.shape == Shape.ELLIPSIS:
        ctx = sc.ellipse(ctx, ctr, p)
    else:
        ctx = sc.random_shape(ctx, ctr, p)

    tmp = np.frombuffer(img.get_data(), np.uint8)
    tmp.shape = [w, h, 4]
    tmp = tmp[:, :, 0:3]
    if FLAGS.resize:
        (h, w) = FLAGS.size
        tmp = image_resize(tmp, w, h)
    cv2.imwrite(dst, tmp)
    # img.write_to_png(dst)
    ctx.restore()
    ctx = shapeCreator.clearContext(ctx, [255, 255, 255])

