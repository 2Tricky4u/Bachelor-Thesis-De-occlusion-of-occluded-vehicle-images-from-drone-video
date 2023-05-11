import os
import cv2
import numpy as np

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
    original_path = "./v_patches_square/"
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
        line = str(img.shape)
        f.write(line + '\n')
        translated_img = cv2.resize(img, (128, 128))
        center = translated_img[51:77, 51:77,:]
        if ratio_of_green(translated_img) < 0.05 and ratio_of_green(center) < 0.01:
            cv2.imwrite(next_path, translated_img)
        else:
            print(ratio_of_green(img))
            print(ratio_of_green(center))
            cv2.imwrite(os.path.join("./green/", file), translated_img)
    f.close()

if __name__ == '__main__':
    print(" - Start the creation of the dataset with the following parameters: - ")
    main()
#%%
