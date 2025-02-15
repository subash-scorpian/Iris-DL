import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Load the saved model
model = keras.models.load_model("iris_model.h5")


# Define class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Custom Header and Description
st.markdown("""
    <style>
    .main-header {
        font-size:36px;
        color:#4CAF50;
        text-align:center;
        font-weight: bold;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        color: black;
        text-align: center;
        padding: 10px;
        background-color: #E0E0E0;
    }
    </style>
    <h1 class='main-header'>🌼 Iris Flower Classification 🌼</h1>
    <p style='text-align: center;'>Predict the type of Iris flower based on its features.</p>
""", unsafe_allow_html=True)


# Streamlit App Layout
st.title("Iris Flower Classification")
st.write("Enter the flower's features below:")

# User Inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Make Prediction
if st.button("🌼 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = class_names[np.argmax(prediction)]
    st.success(f"🌸 Predicted Class: **{predicted_class}**")

st.markdown("""
    <div class='footer'>
         ❤️
    </div>
""", unsafe_allow_html=True)   
