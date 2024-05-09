import cv2
import numpy as np
import pandas as pd


def read_img(file_path):
    img_arr = cv2.imread(file_path)
    return cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB)


def get_mean_and_std(files):
    global_mean = 0
    global_var = 0

    for img in files:
        img_arr = read_img(img) / 255
        global_mean += img_arr.reshape(-1, 3).mean(axis=0)

    global_mean /= len(files)

    for img in files:
        img_arr = read_img(img) / 255
        global_var += ((img_arr.reshape(-1, 3) - global_mean) ** 2).mean(axis=0)

    global_var /= len(files)
    global_std = np.sqrt(global_var)

    return global_mean, global_std


def get_statistics(arr):
    return pd.DataFrame(arr.reshape(-1, 3), columns=["R", "G", "B"]).describe()


def denormalize(img, mean, std):
    img = img.numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = img.clip(0, 1)
    return img
