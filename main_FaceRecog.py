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
#import multiprocessing as mp

album_id= '44'
test_image_path = '/Users/arya/Desktop/test_images'
database_path = '/Users/arya/Desktop/database_images'
#print('\nFor album id : ', album_id, '----------------')
test_list = test.get_imagelist_from_path(test_image_path)
#print('Containing image ids(',str(len(image_list)),'): \n', str(image_list))
  
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
#vector_x.columns=['FD 1']
imageidx = database['IMAGE ID']
vector_y = [] ##for test images' face descriptors
imageidy = [] ##for storing image ids from the test images
vector_y = pd.DataFrame(test_images['FACE DESCRIPTORS']) 
imageidy = test_images['IMAGE ID']
vector_z = pd.concat([vector_x,vector_y],axis=1) 
#vector_x = vector_x.drop("new",axis=1)
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
#vector_z=vector_z.reset_index()   
#vector_z['subtract'] = vector_z.apply(lambda vector_z: np.array(vector_z['FD 1'][0]) - np.array(vector_z['FD 2'][0],axis=1))
#vector_z['subtract'] = vector_z['FD 1'][0].subtract(vector_z['FD 2'][0]) 
#vector_x.set_index('FACE DESCRIPTORS')[0].subtract(vector_y.set_index('FACE DESCRIPTORS')[0])
#vector_x['new'] = vector_x.apply(lambda vector_x: sum(vector_x['FACE DESCRIPTORS'][0] - vector_y['FACE DESCRIPTORS'][0])**2,axis=1) #for i in range(0,len(database))))
#vector_x['new'] = vector_x.apply(lambda x:vector_x.loc['FACE DESCRIPTORS'][0] - vector_y['FACE DESCRIPTORS'][0]**2),axis=1) #for i in range(0,len(database))))
#temp = np.size((vector_z)['database'][0])
#i = range(0,len(test_images))
#j = range(0,len(database))
#dim = range(temp)
#new_df = [] 

#x ={} 
#new_df = list(map(lambda j,dim: np.sum(np.square(vector_z['database'][j][0][dim] - vector_z['test_images'][0][0][0])),j,dim))

### trial
"""
new = []
dat_count, test_count = 0, 0
end = 0
for test_image in vector_z['test_images'][test_count]:
    for data in vector_z['database'][dat_count]:
        while end != temp+1:
            func = lambda k : sum((data[k] - test_image[k])**2)

### trial 2
array_1 = []
array_2 = []        
for i in range(len(database)):
    d = vector_z['database'][i]
    t = vector_z['test_images'][0]
    array_1 = np.append(array_1,d)
    array_2 = np.append(array_2,t)
func = lambda x: array_1 
"""

###final 
"""
def euclidean_distance(data,test_image): 
    #tempX = np.array(vector_z['database'][data])
    data = np.array(data[0])
    #print(np.size(data))
    #print(data[1])
    #tempidx = imageidx[j]
    tempY = test_image
    #tempidy = imageidy[i]
    if(np.size(data)!= np.size(test_image)):
        print("Vectors must be of the same dimensions") 
    else: 
        return sum((data[0][dim] - tempY[0][dim])**2 for dim in range(np.size(data)))#,tempidx,tempidy
#new = new.drop("subtract",axis=1)
def minimum_distance():
    mini = []
    new = pd.DataFrame(vector_z['database'])

#tempidx = list(imageidx)
#tempidy = list(imageidy)    
    for img in range(len(test_images)):
        test_image = np.array(vector_z['test_images'][img])
#l = range(len(database))
#sumlist = []
#i=0 
#sumlist = list(map(lambda x: euclidean_distance(x,test_image), l))
        new['difference'] = new.apply(lambda x: euclidean_distance(np.array(x),test_image),axis=1)
#print("Distance between image id {} and image id {} is {}".format(tempidx,tempidy,sumlist[i])) 
#i=i+1
        minimum = min(new['difference'])
        if(minimum <=0.32):
            mini = np.append(mini,minimum) 
            #print("The minimum distance is = {}".format(mini))
            #print("__________________________________________")
            #print("\n")
        else:
            return 0
            #print("No match found")
            #print("__________________________________________")
            #print("\n")
    return mini
start = time.time()
mini = minimum_distance()
print("The minimum distance is = {}".format(mini))
end = time.time()
print("Time taken for lambda method:{}".format(end-start))
"""
###final bole toh ekdum final 
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