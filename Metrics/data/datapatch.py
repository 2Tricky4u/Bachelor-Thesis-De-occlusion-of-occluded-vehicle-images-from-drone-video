import glob
import random
import PIL
import os
import os.path
from PIL import Image
import cv2
import numpy as np

models = ["RP", "AOT"]
masks = ["small", "big", "GT"]
gt = False
for model in models:
   for mask in masks:
      tmp = os.path.join(os.getcwd(), (model + "_" + mask))
      if(mask == "GT" ):
         if (gt == False):
            tmp = os.path.join(os.getcwd(), mask)
            gt = True
         else:
            break
      print(tmp)
      for count, f in enumerate(glob.glob(os.path.join(tmp, '*.png')), 1):
         imga = cv2.imread(f)
         img = imga[:, :, :3]
         cropped = img[70:186, 70:186]
         num = "\\" + str(count) + ".png"
         print(tmp+"\\patch"+num)
         cv2.imwrite(tmp+"\\patch"+num, cropped)
         #with open(os.path.join(os.getcwd(), f), 'r') as f:



