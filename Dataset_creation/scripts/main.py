
# import cairo
import argparse
import os.path
import pprint

import numpy as np
from splitter import split
from utils import randomShape
from maskCreator import create_masks

# from paint_studio.painter import drawStimuli
# from fio.saver            import saveData


# PARAMETERS ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# general -------------------------------------------------------------------------------

parser.add_argument("--data_path", "-in", type=str, nargs='+', default='./',
                    help="Path to data")

parser.add_argument("--train_data_output", "-tro", type=str, nargs='+', default='./train/',
                    help="Save path for train data (default: ./train/)")

parser.add_argument("--test_data_output", "-teo", type=str, nargs='+', default='./test/',
                    help="Save path for test data (default: ./test/)")

parser.add_argument("--train_ratio", "-r", type=float, nargs='?', default=0.7,
                    help="Train ratio (default: 0.7) means splitting data in 70 % train and 30 % test")

parser.add_argument("--hidden_ratio", '-hr', type=float, nargs='?', default=0.5,
                    help="The percentage of the origanl image to be filled")
# canvas --------------------------------------------------------------------------------
parser.add_argument('--size', type=int, nargs='+', default=[128, 128],
                    help='Size of Images (default: 128x128px)')

parser.add_argument('--bgcol', type=float, nargs='+', default=[255., 255., 255.],
                    help='Background Colour of Images (RGB,default: 255 255 255)')

parser.add_argument('--maskcol', type=float, nargs='+', default=[0., 0., 0.],
                    help='Mask Colour (RGB,default: 0 0 0)')

# randomness------------------------------------------------------------------------------
parser.add_argument('--rec', type=int, nargs='?', default=1,
                    help='Weight of rectangles mask proportion')

parser.add_argument('--poly', type=int, nargs='?', default=1,
                    help='Weight of polygone mask proportion')

parser.add_argument('--star', type=int, nargs='?', default=1,
                    help='Weight of star mask proportion')

parser.add_argument('--circle', type=int, nargs='?', default=1,
                    help='Weight of circle mask proportion')

parser.add_argument('--ellipse', type=int, nargs='?', default=1,
                    help='Weight of ellipse mask proportion')

parser.add_argument('--random', type=int, nargs='?', default=1,
                    help='Weight of random mask proportion')

# output option-----------------------------------------------------------------------------
parser.add_argument('--resize', type=bool, nargs='?', default=False,
                    help='The image is being fit in the size without ratio change')

FLAGS, _ = parser.parse_known_args()  # ignore unspecified args


def main(argv=None):
    split(FLAGS.data_path[0], FLAGS.train_data_output[0], FLAGS.test_data_output[0], FLAGS.train_ratio)
    create_masks(FLAGS)
    print("\nThe dataset was successfully created!" +
          "\nThe train dataset is at location: " + os.path.abspath(FLAGS.train_data_output[0]) +
          "\nThe test dataset is at location: " + os.path.abspath(FLAGS.test_data_output[0]))


if __name__ == '__main__':
    print(" - Start the creation of the dataset with the following parameters: - ")
    pprint.pprint(FLAGS.__dict__)
    main()
