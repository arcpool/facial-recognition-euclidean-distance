#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:40:55 2019

@author: arya
A program to return the Image IDs, Face arrays, and Face descriptors of database images
as a dataframe to the main program.
"""

import dlib
import os
import glob
import cv2
import numpy as np

predictor_path = '/Users/arya/Desktop/dlib_stuff/shape_predictor_5_face_landmarks.dat' 
face_rec_model_path = '/Users/arya/Desktop/models_for_ML/dlib_face_recognition_resnet_model_v1.dat'
 
#detector classes 
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def get_imagelist_from_path(album_path):
    image_list = [image_loc for image_loc in glob.glob(os.path.join(album_path, '*.*'))]
    return image_list

def dlib_on_image(image_path):
    img = cv2.imread(image_path,0) ###inputting the image as grayscale
    r = 1500.0/img.shape[1]   
    dim = (1500, int(img.shape[0]*r))  
    image = cv2.resize(img,dim,interpolation = cv2.INTER_AREA) 
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 
    img_result = {}
       
    #FACE DETECTION
    dets= detector(image)  
    if len(dets)!=0: 
        descriptor_list = [] 
        faceimg_list = [] 
        for k,d in enumerate(dets): #print (k,d) #0 [(3499,1449) (3767,1717)]  
            shape = sp(image,d) #dlib.full_object_detection
            s = shape.rect 
            face_descriptor = facerec.compute_face_descriptor(image, shape) #dlib.vector
            faceimg = image[s.top():(s.top()+s.height()),s.left(): (s.left()+s.width())]
            descriptor_list.append(face_descriptor)
            faceimg_list.append(faceimg)
        descriptors = [np.stack(i) for i in descriptor_list] 
        face_arrays = [np.stack(i) for i in faceimg_list]  
        image_id = image_path.split('/')[-1] 
        img_result = {'IMAGE ID': image_id.split('-')[2], 'FACE ARRAYS': face_arrays,
                      'FACE DESCRIPTORS': descriptors}
    else:
        image_id = image_path.split('/')[-1]
        img_result = image_id.split('-')[2] 
    
    return img_result

   
 
