#svm-digits
Recognizing images of hand-written digits using Support Vector Machines. Each datapoint is a 8x8 image of a digit. 

#Convert it to 2D array using [[-1]] or data[-1].reshape(1,-1)

#1D array Shape is (64,) It has no rows and columns since it is 1d array
print(digits.data[-1])

#2D array Shape is 1row 64columns (1,64)
print(digits.data[[-1]])

#Result same as above, 1 row and number of columns same as number of elements inside 2d array. Shape is 1row 64columns (1,64)
print(digits.data[-1].reshape(1, -1))

#Number of rows same as number of elements inside 2d array, 1 column. Shape is 64rows 1column (64,1)
print(digits.data[-1].reshape(-1, 1))

Better to use reshape

.shape shows how many columns and rows are there 

*Useful Sources*

https://stackoverflow.com/questions/12760797/imshowimg-cmap-cm-gray-shows-a-white-for-128-value
https://processing.org/tutorials/2darray/
https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
https://www.geeksforgeeks.org/numpy-reshape-python/
https://note.nkmk.me/en/python-numpy-reshape-usage/
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
