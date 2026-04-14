import pandas as pd
import streamlit as st
from pickle import load

st.title("Gender Prediction App")
st.sidebar.header("Input Parameters")

def user_input():
    weight_kgs = st.sidebar.number_input("Weight")
    height_cm = st.sidebar.number_input("Height")
    data = {'weight_kgs': weight_kgs, 'height_cm': height_cm}
    features = pd.DataFrame(data, index=[0])
    return features

df= user_input()
st.subheader("User Input Parameters")
st.write(df)

model = load(open(r'c:\Users\elang\python\gender prediction\gender_int.pkl', 'rb'))
prediction=model.predict_proba(df)
st.subheader("Prediction result")
st.write("female"if prediction[0][1]>0.5 else "male")
st.subheader("Prediction")
st.write(prediction)

