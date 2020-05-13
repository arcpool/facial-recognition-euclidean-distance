# Facial Recognition Using Euclidean Distance

main_FaceRecog.py - Main program to perform facial recognition by using Euclidean distance on two 128D vectors(facial descriptors). The program determines the minimum euclidean distance of each input images with each of the database images and returns it as the best recognized face. I used two Python lambda functions on data-frames for iteration instead of ‘for loops’ since Python lambda functions are more efficient.

testimages_face_descriptors.py - A program to return the Image IDs, Face Arrays, and Face descriptors of test images as a dataframe to the main_FaceRecog.py program. 

Database.py - A program to return the Image IDs, Face Arrays, and Face descriptors of database images as a dataframe to the main_FaceRecog.py program. 

