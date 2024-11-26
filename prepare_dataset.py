# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:51:29 2023

@author: KH_Sulemani

Dataset from: http://press.liacs.nl/mirflickr/mirdownload.html

"""
import pywt
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = "data" 

for img in os.listdir( train_dir + "/original_images"):
    img_array = cv2.imread(train_dir + "/original_images/" + img)
    
    img_array = cv2.resize(img_array, (256,256))
    lr_img_array = cv2.resize(img_array,(64,64))
    cv2.imwrite(train_dir+ "/hr_images/" + img, img_array)
    cv2.imwrite(train_dir+ "/lr_images/"+ img, lr_img_array)