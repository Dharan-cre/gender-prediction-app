import pandas as pd
import streamlit as st
import os
from pickle import load

st.title("Gender Prediction App")
st.sidebar.header("Input Parameters")

def user_input():
    weight_kgs = st.sidebar.number_input("Weight")
    height_cm = st.sidebar.number_input("Height")
    data = {'weight_kgs': weight_kgs, 'height_cm': height_cm}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()

st.subheader("User Input Parameters")
st.write(df)

# ✅ FIXED MODEL PATH (important)
model_path = os.path.join(os.path.dirname(__file__), "gender_int.pkl")
model = load(open(model_path, 'rb'))

prediction = model.predict_proba(df)

st.subheader("Prediction result")
st.write("female" if prediction[0][1] > 0.5 else "male")

st.subheader("Prediction probability")
st.write(prediction)