import streamlit as st
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('Car_prediction_LR_model.pkl','rb'))
df = pd.read_csv('Cleaned car.csv')

st.title('Car price Predictior')
# print(df.head())

# name
name = st.selectbox("Car Brand",df['name'].unique())

# company
company = st.selectbox("Company",df['company'].unique())
# year
year = st.number_input("Year of purchase",format='%i',max_value=2022,min_value=2002)
# Price
# kms_driven
kms_driven = st.number_input("Kilometers Driven",format='%i',min_value=df['kms_driven'].min())
# fuel_type
fuel_type = st.selectbox("Fuel",df['fuel_type'].unique())

if st.button("Predict"):
    query = pd.DataFrame([[name,company, year,kms_driven,fuel_type]], columns=["name", "company", "year", "kms_driven", "fuel_type"])
    result = round(pipe.predict(query)[0])
    if result<0:
        st.title("The model says this car is not sellable.")
    else:
        st.title("Selling Price should be: "+str(result)+" INR.")