import cv2
import skimage.exposure
import cairo
import math
import numpy as np
from scipy.special import binom
from numpy.random import default_rng

## From https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)

"""Create a beziers curves as coords

:param points: number of points
:param num: precision for the discrete value
:returns: the coords of the curve
"""


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


"""
Class to represent a segment
"""


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    """Segment class function to generate intermediate points

    :param r: self.r
    :returns: the coords of the segment
    """

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


"""Returns the segments and the curve given by points

:param points: the points to represent a curve
:param kw: kw
:returns: (the segments composing the curve, the curve)
"""


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


"""Function used to sort the as of bezier curve

:param p: the points
:returns: sorted array
"""


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


"""Get the bezier curve out of rad and edgy

:param a: a
:param rad: the rad parameter
:param edgy: the edgy parameter
:returns: coords x, coords y, a
"""


def get_bezier_curve(a, rad=0.2, edgy=0.):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


"""Create some random points that are scaled

:param n: number of coords
:param scale: the scale factor
:param mindst: mindst
:param rec: rec
:returns: the array of random points scaled
"""


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


# Inspired from https://github.com/TimoFlesch/2D-Shape-Generator
# Canvas creation
"""Helper function to make a surface for cairo canvas

:param ssisze: size of canvas
:returns: img representing the surface
"""


def makeSurface(ssize):
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, ssize[0], ssize[1])
    return img


"""Helper function to make a contex for cairo canvas 

:param img: the image of the canvas (surface)
:returns: the context of the image
"""


def makeContext(img):
    return cairo.Context(img)


"""Helper function to make a canvas for cairo lib

:param imsize: size of the image canvas
:param bgcol: color of the background
:returns: the image canvas, the context
"""


def makeCanvas(imsize, bgcol):
    img = makeSurface(imsize)
    ctx = makeContext(img)
    ctx.set_source_rgb(bgcol[0], bgcol[1], bgcol[2])
    ctx.paint()
    return img, ctx


"""Helper function to clear the context of the cairo canvas

:param ctx: the canvas context
:param col: color to be filled
:returns: the cleared context
"""


def clearContext(ctx, col):
    ctx.set_source_rgb(col[0], col[1], col[2])
    ctx.paint()
    return ctx


"""Helper function to translate a shape 

:param ctx: the context of the canvas
:param dx: x axis displacement (width)
:param dy: y axis displacement (height)

:returns: the canvas context
"""


# Transformation
def translateShape(ctx, dx=0, dy=0):
    ctx.translate(dx, dy)
    return ctx


"""Helper function to rotate a shape 

:param ctx: the context of the canvas
:param angle: the angle to rotate the shape (degree)

:returns: the canvas context
"""


def rotateShape(ctx, angle):
    ctx.rotate(math.radians(angle))
    return ctx


"""Helper function to scale a shape 

:param ctx: the context of the canvas
:param scale: the scale factor
:returns: the canvas context
"""


def scaleShape(ctx, scale):
    ctx.scale(scale[0], scale[1])
    return ctx


"""Helper function to colourise a shape 

:param ctx: the context of the canvas
:param newCol: the color to apply to the shape

:returns: the canvas context
"""


def colouriseShape(ctx, newCol):
    ctx.set_source_rgb(newCol[0], newCol[1], newCol[2])
    return ctx


# Shapes
"""Helper function to draw a rectangle 

:param ctx: the context of the canvas
:param size: the size used for the shape
:param rectRatio: the ratio used to scal edge of the rectangle

:returns: the canvas context
"""


def drawRect(ctx, size, rectRatio):
    ctx.rectangle(0 - (size / rectRatio[0]) / 2, 0 - (size / rectRatio[1]) / 2, size / rectRatio[0],
                  size / rectRatio[1])
    return ctx


"""Helper function to draw a polygone

:param ctx: the context of the canvas
:param size: the size used for the shape
:param numVertices: the number of vertices

:returns: the canvas context
"""


def drawPolygon(ctx, size, numVertices=4):
    circ = 2 * math.pi
    angle = circ / numVertices
    coords = [(math.sin(angle * ii) * size, math.cos(angle * ii) * size) for ii in range(numVertices)]
    for ii in coords:
        ctx.line_to(ii[0], ii[1])
    return ctx


"""Helper function to draw a star 

:param ctx: the context of the canvas
:param outerRadius: the radius for the outer spike
:param innerRadius: the radius for the inner edge
:param numVertices: the number of spike

:returns: the canvas context
"""


def drawStar(ctx, outerRadius, innerRadius, numVertices=5):
    circ = 2 * math.pi
    angle = circ / (numVertices * 2)
    coords = []
    for ii in range(numVertices * 2):
        r = innerRadius if ii % 2 else outerRadius
        coords.append([math.sin(angle * ii) * r, math.cos(angle * ii) * r])
    # coords = [[sum(translate_op) for translate_op in zip(coordPair,centre)]for coordPair in coords]

    for ii in coords:
        ctx.line_to(ii[0], ii[1])
    # ctx.close_path()
    return ctx


"""Helper function to draw a random shape 

:param ctx: the context of the canvas
:param size: the max size of the shape (square)

:returns: the canvas context
"""


def drawRandom(ctx, size):
    rad = 0.2
    edgy = 0.05
    a = get_random_points(n=7, scale=size)
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    coords = zip(x - size / 2, y - size / 2)

    for ii in coords:
        ctx.line_to(ii[0], ii[1])
    # ctx.close_path()
    return ctx


"""Helper function to draw a circle

:param ctx: the context of the canvas
:param size: the radius of the circle

:returns: the canvas context
"""


def drawCircle(ctx, size):
    ctx.arc(0, 0, size, 0, math.pi * 2)
    return ctx


"""Helper function to draw a star 

:param ctx: the context of the canvas
:param centre: the center of the ellipsis
:param size: the size of the radius
:param scale: the scale factor for the first radius of the ellipsis and the second one

:returns: the canvas context
"""


def drawEllipse(ctx, centre, size, scale):
    ctx.scale(scale[0], scale[1])
    ctx = drawCircle(ctx, size)
    ctx.scale(1, 1)
    ctx.translate(0, centre[1])
    return ctx


"""Helper function to draw random shapes with help of noise (no control of center nor number)

:param w: the width
:param h: the height

:returns: a cv2 morphology ex
"""


def random_shapes(w, h):
    seedval = 1
    rng = default_rng(seed=seedval)

    # define image size
    width = w
    height = h

    # create random noise image
    noise = rng.integers(0, 255, (height, width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (127, 127), sigmaX=30, sigmaY=30, borderType=cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


"""Helper function to prepare a shape being draw on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the shape

:returns: the canvas context
"""


def prepare(ctx, ctr, p):
    ctx = translateShape(ctx, ctr[0], ctr[1])
    ctx = translateShape(ctx, p.trX, p.trY)
    ctx = rotateShape(ctx, p.rota)
    ctx = scaleShape(ctx, p.scale)
    ctx = colouriseShape(ctx, p.mask_color)
    return ctx


"""Helper function to put a rectangle on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the rectangle

:returns: the canvas context
"""


def rectangle(ctx, ctr, p):
    prepare(ctx, ctr, p)
    ctx = drawRect(ctx, p.size, p.rect_ratio)
    ctx.fill()
    return ctx


"""Helper function to put a polygone on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the polygone

:returns: the canvas context
"""


def poly(ctx, ctr, p):
    prepare(ctx, ctr, p)
    ctx = drawPolygon(ctx, p.size, p.vertices)
    ctx.fill()
    return ctx


"""Helper function to put a star on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the star

:returns: the canvas context
"""


def star(ctx, ctr, p):
    prepare(ctx, ctr, p)
    ctx = drawStar(ctx, p.size, p.size2, numVertices=p.branch)
    ctx.fill()
    return ctx


"""Helper function to put a circle on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the circle

:returns: the canvas context
"""


def circle(ctx, ctr, p):
    prepare(ctx, ctr, p)
    ctx = drawCircle(ctx, p.size)
    ctx.fill()
    return ctx


"""Helper function to put an ellipsis on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the ellipsis

:returns: the canvas context
"""


def ellipse(ctx, ctr, p):
    prepare(ctx, ctr, p)
    ctx = drawEllipse(ctx, ctr, p.size, p.ellipse_ratio)
    ctx.fill()
    return ctx


"""Helper function to put a random shape on canvas

:param ctx: the context of the canvas
:param ctr: the center of the canvas
:param p: the properties object of the random shape

:returns: the canvas context
"""


def random_shape(ctx, ctr, p):
    prepare(ctx, (ctr[0], ctr[1]), p)
    ctx = drawRandom(ctx, p.size)
    ctx.fill()
    return ctx


"""Helper function to return the area of the rectangle

:param size: base size
:param ratio: ratio for size length


:returns: area of the rectangle
"""


def area_rec(size, ratio):
    (l1, l2) = size * ratio
    return l1 * l2


"""Helper function to return the area a polygone (estimated)

:param size: base size

:returns: area of a polygone
"""


def area_poly(size):
    return math.pi * (size ** 2)


"""Helper function to return the area of a star (estimated)

:param size: outerRadius


:returns: area of a star
"""


def area_star(size):
    return math.pi * (size ** 2) / 2


"""Helper function to return the area of a circle

:param size: Radius


:returns: area of a circle
"""


def area_circle(size):
    return math.pi * (size ** 2)


"""Helper function to return the area of an ellipsis

:param size: base size
:param ratio: ratio scale factor for the base size


:returns: area of an ellipsis
"""


def area_ellipse(size, ratio):
    (a, b) = size * ratio
    return math.pi * a * b


"""Helper function to return the area of a random shape (estimated)

:param size: max size of a square vertex delimitation


:returns: area of the random shape (badly estimated)
"""


def area_random(size):
    return (size ** 2) * 3 / 4


"""Helper function to try to predict the size for the area of a rec (estimated)

:param dim: dimension of the total image
:param rec_ratio: ratio of the rectangle shape
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_rec(dim, rec_ratio, ratio):
    (a, b) = dim
    (r1, r2) = rec_ratio
    return int(round(math.sqrt(ratio * a * b / (r1 * r2))))


"""Helper function to try to predict the size for the area of a polygone (estimated)

:param dim: dimension of the total image
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_poly(dim, ratio):
    (a, b) = dim
    return int(round(math.sqrt(ratio * a * b / math.pi)))


"""Helper function to try to predict the size for the area of a star (estimated)

:param dim: dimension of the total image
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_star(dim, ratio):
    (a, b) = dim
    return int(round(math.sqrt(2 * ratio * a * b / math.pi)))


"""Helper function to try to predict the size for the area of a circle (estimated)

:param dim: dimension of the total image
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_circle(dim, ratio):
    return size_poly(dim, ratio)


"""Helper function to try to predict the size for the area of an ellipsis (estimated)

:param dim: dimension of the total image
:param ellipsis_ratio: ratio of the ellipsis radius shape
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_ellipse(dim, ellipse_ratio, ratio):
    (a, b) = dim
    (r1, r2) = ellipse_ratio
    return int(round(math.sqrt(ratio * a * b / (math.pi * r1 * r2))))


"""Helper function to try to predict the size for the area of a random shape (estimated)

:param dim: dimension of the total image
:param ratio: the ratio of the total image to hide with the shape


:returns: size to use to create the shape
"""


def size_random(dim, ratio):
    (a, b) = dim
    return int(round(math.sqrt(ratio * a * b)))
