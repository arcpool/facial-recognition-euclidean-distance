#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:42:36 2019

@author: arya
Main program to perform facial recognition by using eucledian distance on two 128d vectors
 --> facial descriptors 
 --> use of lambda functions 

"""
import testimages_face_descriptors as test 
import Database as database
import pandas as pd
import time
import numpy as np

album_id= '44'
test_image_path = '/Users/arya/Desktop/test_images'
database_path = '/Users/arya/Desktop/database_images'
test_list = test.get_imagelist_from_path(test_image_path)
  
print('starting test image processing....')
call_ip = time.time()
result = [test.dlib_on_image(image) for image in test_list]
return_ip = time.time()
print("Time taken for test image processing : ", return_ip -  call_ip)

###storing test images' face descriptors in an dataframe
test_images = pd.DataFrame(result)   

#####for database images
data_list = database.get_imagelist_from_path(database_path) 
print('starting database image processing....')
call_ip = time.time()
result = [database.dlib_on_image(image) for image in data_list]
return_ip = time.time()

print("Time taken for database image processing : ", return_ip -  call_ip)
print("__________________________________________")
print("\n")
###storing database images' face descriptors in an dataframe
database = pd.DataFrame(result)  

###extracting face descriptors(vectors) from both the dataframes
vector_x = [] ##for database's face descriptors
imageidx = [] ##for storing image ids from the database
vector_x = pd.DataFrame(database['FACE DESCRIPTORS'])
imageidx = database['IMAGE ID']
vector_y = [] ##for test images' face descriptors
imageidy = [] ##for storing image ids from the test images
vector_y = pd.DataFrame(test_images['FACE DESCRIPTORS']) 
imageidy = test_images['IMAGE ID']
vector_z = pd.concat([vector_x,vector_y],axis=1) 
vector_z = vector_z.fillna(0)
vector_z.columns= ['database', 'test_images'] 
###calculating the euclidean distance between two vectors 
result={}

def euclidean_dist(i,j): 
    tempX = np.array(vector_x['FACE DESCRIPTORS'][j])
    tempidx = imageidx[j]
    tempY = np.array(vector_y['FACE DESCRIPTORS'][i])
    tempidy = imageidy[i]
    if(np.size(tempX)!= np.size(tempY)):
        print("Vectors must be of the same dimensions") 
    else: 
        return sum((tempX[0][dim] - tempY[0][dim])**2 for dim in range(np.size(tempX))), tempidx,tempidy
start = time.time()
for i in range(0,len(test_images)):
    result = {}
    for j in range(0,len(database)):  
        match, tempidx, tempidy = euclidean_dist(i,j)
        print("Distance between image id {} and image id {} is {}".format(tempidx,tempidy,match)) 
        if match <=0.32:
            result.update({tempidx:match}) ##to store all the distances of the faces matched
            print("matched")  
        else: 
            print("not matched")
    if(bool(result) == True ):
        min_val = min(result.values()) ##variable to store the minimum distance 
        for k in result.keys():
            if min_val == result[k]: 
                print("The minumum distance is = {} with image {}".format(min_val, k))
                print("__________________________________________")
                print("\n")
    else:
        print("No match found")
        print("__________________________________________")
        print("\n")
end = time.time()

print("Time taken for two 'for loops':{}".format(end - start)) 
print("__________________________________________")
print("\n")

def eucli_dist(face_des,db,mat): 
    db = db[0]
    if(np.size(db)!= np.size(face_des)):
        print("Vectors must be of the same dimensions") 
    else: 
        mat = np.append(mat,sum((db[0][dim] - face_des[0][dim])**2 for dim in range(np.size(db))))
        return mat 

def func(face_des, vector_x,match):
    mat = []
    face_des = face_des[0]
    match = np.append(match,vector_x.apply(lambda z: eucli_dist(face_des, np.array(z),mat),axis = 1))
    mini = min(match)
    if(mini<=0.32): 
        return mini
    else:
        return "not found"  

difference_df = vector_y
match = []
start1 = time.time()
difference_df['dif'] = difference_df.apply(lambda x: func(np.array(x), vector_x,match),axis = 1) ##calling func passing the fd of test image and full db
final1 = time.time()
print("Time taken for lambda method:{}".format(final1-start1))
