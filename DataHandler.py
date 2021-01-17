"""
A program which prepares the dataset for use.
"""

import os
import random
import cv2
import numpy as np


class DataHandler:

    def __init__(self):
        self.DATA_FP = os.path.abspath(os.path.join(os.curdir, 'data'))

    # get covid image data file paths
    def get_covid_data_fp(self):
        covid_dir = os.path.join(self.DATA_FP, 'CT_COVID')
        fp_list = [os.path.join(covid_dir, fp) for fp in os.listdir(covid_dir)]
        return fp_list

    # get non-covid image data file paths
    def get_non_covid_data_fp(self):
        non_covid_dir = os.path.join(self.DATA_FP, 'CT_NonCOVID')
        fp_list = [os.path.join(non_covid_dir, fp) for fp in os.listdir(non_covid_dir)]
        return fp_list

    # label covid image data with 1s and label non-covid image data with 0s
    def get_all_data_labeled(self, shuffle=False):
        dataset = [(fp, 1.0) for fp in self.get_covid_data_fp()]
        dataset += [(fp, 0.0) for fp in self.get_non_covid_data_fp()]
        if shuffle:
            random.shuffle(dataset)
        return dataset

    # method loads an image from a given filepath and has options to rescale, make the image greyscale, add gaussian
    # noise to the image, and randomly rotate the image 0-3 times 90 degrees
    def load_image(self, fp, resize=False, grayscale=False, add_noise=False, randomly_rotate=False):
        fp = os.path.abspath(fp)
        if not os.path.isfile(fp):
            print("Could not find image at location: ", fp)
        img = cv2.imread(fp)
        if randomly_rotate:
            # randomly rotate the image 90 degrees 0 to 3 times.
            times_to_rotate = np.random.randint(0, 4)
            img = np.rot90(img, times_to_rotate)
        if resize:
            resized_img = img.copy()
            # get which dimension is larger and resize based on that
            # if height value is greater than width value
            if img.shape[0] > img.shape[1]:
                new_height = 299
                new_width = int(np.floor((299 * img.shape[1]) / img.shape[0]))
            else:
                new_width = 299
                new_height = int(np.floor((299 * img.shape[0]) / img.shape[1]))
            resized_img = cv2.resize(resized_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # padding added so both dimensions are 299
            pad_left = int((299 - resized_img.shape[1]) / 2)
            pad_top = int((299 - resized_img.shape[0]) / 2)
            pad_right = int(299 - pad_left - resized_img.shape[1])
            pad_bot = int(299 - pad_top - resized_img.shape[0])
            img = cv2.copyMakeBorder(resized_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                     value=0)
        if add_noise:
            img = np.float32(img)
            mean = 0.0
            var = 1.0
            stdev = var ** 0.5
            gauss_noise = np.random.normal(mean, stdev, (img.shape[0], img.shape[1], img.shape[2]))
            gauss_noise = np.reshape(gauss_noise, (img.shape[0], img.shape[1], img.shape[2]))
            img = img + gauss_noise
            img = cv2.cvtColor(img.astype('float32'), cv2.COLOR_RGB2BGR)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (img.shape[0], img.shape[1]))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
