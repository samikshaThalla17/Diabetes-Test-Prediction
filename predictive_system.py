# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loded_model = pickle.load(open("C:/Users/nikhi/Desktop/sam's/trained_model.sav",'rb'))


input_data = (7,196,90,0,0,39.8,0.451,41)

#changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print("The persion is non-Diabetic")
else:
    print("The person the Diabetic")