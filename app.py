import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

with open('geo_ohe_encoder.pkl', 'rb') as file:
    ohe = pickle.load(file)

with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the categorical features
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))    

#Concatenate the rest of the input data with the one-hot encoded features (numeric first, then one-hot)
input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Reorder columns to match training order
# expected_columns = scaler.feature_names_in_
# input_df = input_df[expected_columns]

# Scale the input data
input_df_scaled = scaler.transform(input_df)

# Predict the output using the trained model
prediction = model.predict(input_df_scaled)
preiction_proba = prediction[0][0]

#Print the prediction result
st.write(f"Churn Probability: {preiction_proba:.2f}")


# Display the prediction result
if preiction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

st.write("Input DataFrame:")
st.write(input_df)
st.write("Input columns:", input_df.columns.tolist())
st.write("Scaler columns:", scaler.feature_names_in_.tolist())
input_df_scaled = scaler.transform(input_df)
st.write("Scaled input:", input_df_scaled)
