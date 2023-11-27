import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Real Estate Price Prediction")
filename = 'model.pkl'

with open(filename,'rb') as file:
    model = pickle.load(file)

df = pd.read_csv("data.csv")

st.header("Enter the details of your house to get price")

sqft = st.slider('Total Square Feet', min_value=100, max_value=10000, value=1000, step=100)
bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
bhk = st.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2)
balcony = st.number_input('Number of Balcony', min_value=1, max_value=5, value=2)

def predict_price(sqft,bath,bhk,balcony):
    inp = np.array([sqft,bath,bhk,balcony]).reshape(1,-1)
    pred = model.predict(inp)
    return pred

if st.button('Predict Price'):
    predicted_price = predict_price(sqft,bath,bhk,balcony)
    st.subheader(f"The predicted price of your house is : {round(predict_price(sqft,bath,bhk,balcony)[0],2)} lakhs")
