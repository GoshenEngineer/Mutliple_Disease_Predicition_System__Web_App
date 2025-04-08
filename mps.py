# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 21:09:08 2025

@author: Admin
"""
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

#loading the saved models
#for the diabetes model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
#for the heart_disease model
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
#for the parkinsons model
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
                             

#sidebar for navigate
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System using ML',
                           
                           
                           ['Diabetes Prediction', 'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons = ['activity','heart','person'],
                           
                           default_index= 0)

#Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    #page title
    st.title('Diabetes Prediction using ML')
    
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:    
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:    
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:    
        Insulin =st.text_input('Insulin level')
    with col3:    
        BMI = st.text_input('BMI value')
    with col1:    
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:    
        Age = st.text_input('Age of the Person')
    
    # code for prediction
    diab_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array =np.asarray([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)

        diab_prediction =  diabetes_model.predict(input_data_reshaped)
        if(diab_prediction[0]== 0):
           diab_diagnosis = 'The person is not having diabetes'
        else:
           diab_diagnosis = 'The person is having diabetes'
    st.success(diab_diagnosis)
if (selected == 'Heart Disease Prediction'):
    
    #page title
    st.title('Heart Disease Prediction using ML')
  
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input('Age')
    with col2:
        Sex = st.text_input('Sex')
    with col3:    
        trestbps = st.text_input('Resting blood pressure (in mm Hg on admission to the hospital')
    with col1:    
        chol = st.text_input('Serum cholestoral in mg/dl')
    with col2:    
        fbs =st.text_input('(fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)')
    with col3:    
        restecg  = st.text_input('Resting electrocardiographic results')
    with col1:    
        thalach = st.text_input('Maximum heart rate achieved')
    with col2:    
        exang = st.text_input('Exercise induced angina (1 = yes; 0 = no)')
    with col3:
        oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    with col1:
        slope =  st.text_input('The slope of the peak exercise ST segment')
    with col2:
        ca = st.text_input('Number of major vessels (0-3) colored by flourosopy')
    with col3:
        thal = st.text_input('Types of defects related to a heart condition,: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    # code for prediction
    heart_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array =np.asarray([Age, Sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca,thal])
        
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)

        heart_prediction =  heart_disease_model.predict(input_data_reshaped)
        if(heart_prediction[0]== 0):
           heart_diagnosis = 'The person is not having Heart Disease'
        else:
           heart_diagnosis = 'The person is having Heart Disease'
    st.success(heart_diagnosis)
    
if (selected == 'Parkinsons Prediction'):
    
    #page title
    st.title('Parkinsons Prediction using ML')
    
    
    #getting the input data from user
    #columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo = st.text_input('Average vocal fundamental frequency')
    with col2:
        MDVP_Fhi = st.text_input('Maximum vocal fundamental frequency')
    with col3:    
        MDVP_Flo = st.text_input('Minimum vocal fundamental frequency')
    with col1:    
        MDVP_Jitter_ = st.text_input(' The percentage variation in the fundamental frequency, indicating voice instability.')
    with col2:    
        MDVP_RAP =st.text_input('Relative Average Perturbation, another measure of frequency variation')
    with col3:    
        MDVP_PPQ = st.text_input('Pitch Period Perturbation Quotient')
    with col1:    
        Jitter_DDP  = st.text_input('Jitter (Difference of Differences)')
    with col2:    
        MDVP_Shimmer = st.text_input('Variation in amplitude (loudness) of the voice')
    with col3:
        MDVP_Shimmer_dB = st.text_input('Shimmer in decibels, another measure of amplitude variation')   
    with col1:
        Shimmer_DDA = st.text_input('A measure of amplitude variation over dynamic time')
    with col2:
        HNR = st.text_input('Harmonics-to-Noise Ratio')
    with col3:
        RPDE = st.text_input('Recurrence-based Plotting of Dynamic Entropy')
    with col1:
        DFE = st.text_input('The Detrended Fluctuation Exponent (DFE)')
    with col2:
        spread1 = st.text_input('Fundamental Frequency Variation 1')
    with col3:
        spread2 = st.text_input('Fundamental Frequency Variation 2')
    with col1:
        D2 = st.text_input('Fractal Dimension')
    with col2:
        PPE = st.text_input('Peak-to-Peak Entropy')

    # code for prediction
    park_diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Parkinson Test Result'):
        #converting the input data into numpy arrays
        input_data_as_array =np.asarray([MDVP_Fo,MDVP_Fhi, MDVP_Flo ,MDVP_Jitter_  ,MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer,
                                         MDVP_Shimmer_dB, Shimmer_DDA, HNR, RPDE, DFE, spread1, spread2, D2,PPE])
        
        # reshape the array as we are predicting for one patient
        input_data_reshaped = input_data_as_array.reshape(1,-1)

        parkinson_prediction =  parkinsons_model.predict(input_data_reshaped)
        if(parkinson_prediction[0]== 0):
           park_diagnosis = 'The person is not Parkinson Disease'
        else:
           park_diagnosis = 'The person is having Parkinson Disease'
    st.success(park_diagnosis)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    