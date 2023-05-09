import random
from enum import Enum


class Shape(Enum):
    RECTANGLE = 0,
    POLYGONE = 1,
    STAR = 2,
    CIRCLE = 3,
    ELLIPSIS = 4,
    RANDOM = 5


class randomShape:
    def __init__(self, rec, poly, star, circle, ellipse, random):
        next = rec
        self.rec = range(0, next)
        old = next
        next += poly
        self.poly = range(old, next)
        old = next
        next += star
        self.star = range(old, next)
        old = next
        next += circle
        self.circle = range(old, next)
        old = next
        next += ellipse
        self.ellipse = range(old, next)
        old = next
        next += random
        self.random = range(old, next)
        self.total = rec + poly + star + circle + ellipse + random

    def getShape(self):
        choice = random.choice(range(self.total))
        if choice in self.rec:
            return Shape.RECTANGLE
        if choice in self.poly:
            return Shape.POLYGONE
        if choice in self.star:
            return Shape.STAR
        if choice in self.circle:
            return Shape.CIRCLE
        if choice in self.ellipse:
            return Shape.ELLIPSIS
        if choice in self.random:
            return Shape.RANDOM


#%%
