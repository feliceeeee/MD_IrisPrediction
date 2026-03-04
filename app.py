import streamlit as st # membuat web app interaktif
import joblib # load model dan scaler yang sudah disimpan
import numpy as np
import pandas as pd


# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("artifacts/preprocessor.pkl") # scaler hasil preprocessing
model = joblib.load("artifacts/model.pkl") # model hasil training

def main():
    st.title('Machine Learning Iris Prediction Model Deployment')

    # Add user input components for 5 features
    # jangan lupa set nilai min dan max agar invalid data tidak masuk
    sepal_length = st.number_input('input nilai sepal_length', min_value=0.0, max_value=10.0, value=0.1)
    sepal_width = st.number_input('sepal_width', min_value=0.0, max_value=10.0, value=0.1)
    patal_length = st.slider('patal_length', min_value=0.0, max_value=10.0, value=0.1)
    patal_width = st.slider('patal_width', min_value=0.0, max_value=10.0, value=0.1)
    
    # prediksi, ketika tombol ditekan ambil semua fitru, masukkan ke fungsi make_prediction(), tampilkan hasil 
    if st.button('Make Prediction'):
        features = [sepal_length,sepal_width,patal_length,patal_width] # mengumpulkan fitur
        result = make_prediction(features) # ubah input jadi array, scaling, prediksi, return hasil 
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    # semua yang dilakukan tadi, lalukan juga di inferencing data ini
    input_array = np.array(features).reshape(1, -1) # ubah ke array 2D
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()

