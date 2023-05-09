import math
import shutil
import os
import random
import numpy as np

from collections import Counter

"""List of files of a folder

:param path: the path to the folder to seek files 
:returns: an array of files name
"""
def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

"""Split images on a test and train specified folders
With the specified ratio 

:param path_to_data: the path to the folder to seek images
:param path_to_train_data: the path to the folder where we want to store the training images
:param path_to_test_data: the path to the folder where we want to store the test images 
:returns: nothing
"""
def split(path_to_data, path_to_train_data, path_to_test_data, train_ratio):
    # Files walking and counter
    files = get_files_from_folder(path_to_data)
    counter = len(files)
    print("--Processing " + str(counter) + " images--")
    znb = math.ceil(math.log10(counter))
    # Number of test images
    test_counter = int(np.round(counter * (1 - train_ratio))) + 1
    # Data separation
    test_data = np.array(random.choices(files, k=test_counter))
    train_data = np.array(list((Counter(files) - Counter(test_data)).elements()))

    path_to_original = path_to_data

    # Creates dir if not existing
    if not os.path.exists(path_to_train_data):
        os.makedirs(path_to_train_data)
    if not os.path.exists(path_to_test_data):
        os.makedirs(path_to_test_data)

    # Put images on a gt (Ground truth) folder
    new_train_path = os.path.join(path_to_train_data, "gt/")
    new_test_path = os.path.join(path_to_test_data, "gt/")

    # Creates dir if not existing
    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)
    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    # Copy data
    for i, j in enumerate(range(test_counter)):
        dst = os.path.join(new_test_path, str(i).zfill(znb) + ".jpg")
        src = os.path.join(path_to_original, test_data[j])
        shutil.copy(src, dst)

    for i, j in enumerate(range(counter - test_counter)):
        dst = os.path.join(new_train_path, str(i).zfill(znb) + ".jpg")
        src = os.path.join(path_to_original, train_data[j])
        shutil.copy(src, dst)

