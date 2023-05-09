import math
import shutil
import os
import random
import numpy as np

from collections import Counter

def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

def split(path_to_data, path_to_train_data, path_to_test_data, train_ratio):
    # get dirs
    #_, dirs, _ = next(os.walk(path_to_data))
    #path = os.path.join(path_to_data, dirs[i])

    files = get_files_from_folder(path_to_data)
    counter = len(files)
    print("     Processing " + str(counter) + " images")
    znb= math.ceil(math.log10(counter))
    test_counter = int(np.round(counter * (1 - train_ratio)))+1
    test_data = np.array(random.choices(files, k=test_counter))
    train_data = np.array(list((Counter(files) - Counter(test_data)).elements()))


    #path_to_original = os.path.join(path_to_data, dirs[i])
    #path_to_save = os.path.join(path_to_test_data, dirs[i])
    path_to_original = path_to_data

    #creates dir
    if not os.path.exists(path_to_train_data):
        os.makedirs(path_to_train_data)
    if not os.path.exists(path_to_test_data):
        os.makedirs(path_to_test_data)

    new_train_path = os.path.join(path_to_train_data, "gt/")
    new_test_path = os.path.join(path_to_test_data, "gt/")

    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)
    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    # moves data
    for i,j in enumerate(range(test_counter)):
        #dst = os.path.join(path_to_test_data, test_data[j])
        dst = os.path.join(new_test_path, str(i).zfill(znb)+".jpg")
        src = os.path.join(path_to_original, test_data[j])
        shutil.copy(src, dst)

    for i,j in enumerate(range(counter-test_counter)):
        #dst = os.path.join(path_to_train_data, train_data[j])
        dst = os.path.join(new_train_path, str(i).zfill(znb)+".jpg")
        src = os.path.join(path_to_original,  train_data[j])
        shutil.copy(src, dst)


#split('../small_for_test/car','../train', '../test', float(0.7))
