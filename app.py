import streamlit as st
import pickle
import numpy as np

scalar = pickle.load(open('scaling.pkl', 'rb'))
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Title and input prompt
st.title("Fraudulent Transaction Classifier")

# Define input fields for transaction features (assuming the model uses 5 features for illustration)
input_feature1 = st.number_input("Enter type")
input_feature2 = st.number_input("Enter amount")
input_feature3 = st.number_input("Enter oldbalanceOrg")
input_feature4 = st.number_input("Enter newbalanceOrig")
input_feature5 = st.number_input("Enter oldbalanceDest")
input_feature6 = st.number_input("Enter newbalanceDest")
input_feature7 = st.number_input("Enter isFlaggedFraud")
input_feature8 = st.number_input("Enter origin_bal_change")
input_feature9 = st.number_input("Enter dest_bal_increase")
input_feature10 = st.number_input("Enter HourOfDay")

# Predict button
if st.button('Predict'):
    # Collect input features into an array
    input_features = np.array([input_feature1, input_feature2, input_feature3, input_feature4, input_feature5,input_feature6,input_feature7,input_feature8,input_feature9,input_feature10]).reshape(1, -1)
    
    # Scale the input features
    scaled_input = scalar.transform(input_features)
    
    # Make prediction
    result = model.predict(scaled_input)[0]
    
    # Display result
    if result == 0:
        st.header("This is a fair transaction")
    else:
        st.header("This is a fraudulent transaction")