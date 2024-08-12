# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:59:08 2024

@author: nikhi
"""

import numpy as np
import pickle
import streamlit as st

loded_model = pickle.load(open("C:/Users/nikhi/Desktop/sam's/trained_model.sav",'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):

    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return "The persion is non-Diabetic"
    else:
        return "The person is Diabetic"
    

def main():
    
    
    #giving a title
    st.title("Diabetes Prediction Web App")
    
    
    #getting the input data from user
    
    Pregnancies = st.text_input("Number of Pregnancies:")
    Glucose = st.text_input("Glucose Level:")
    BloodPressure= st.text_input("Blood Pressure Value:")
    SkinThickness = st.text_input("Skin Thinckness Value:")
    Insulin = st.text_input("Insulin Level:")
    BMI = st.text_input("BMI Value:")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value:")
    Age = st.text_input("Age of the Person:")
    
    #code for prediction
    Diagnosis = " "
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        Diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(Diagnosis)
    
    
    
    
if __name__ == "__main__":
    main()

    
    

    
    