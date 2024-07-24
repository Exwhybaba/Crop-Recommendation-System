import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# Load model and preprocessing components
model_path = r"C:\Users\HomePC\Documents\weed_detect\mini_project\model.sav"
with open(model_path, mode='rb') as file:
    loaded_model, encoder, scaler, normalizer = pickle.load(file)

# Function to make predictions
def make_prediction(Nitrogen, Phosphorus, Calcium, temperature, humidity, ph, rainfall):
    data = {
        'Nitrogen': Nitrogen,
        'Phosphorus': Phosphorus,
        'Calcium': Calcium,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
    }

    df = pd.DataFrame(data, index=[0])
    df_array = np.asarray(df)
    reshaped_array = df_array.reshape(1, -1)

    def transform_data(reshaped_array):
        scaled_array = scaler.transform(reshaped_array)
        normalized_array = normalizer.transform(scaled_array)
        return normalized_array

    # Transform data
    transformed_data = transform_data(reshaped_array)

    # Make prediction
    prediction = loaded_model.predict(transformed_data)

    # Inverse transform the predicted labels
    decoded_prediction = encoder.inverse_transform(prediction)
    return decoded_prediction[0]

# Function to encode image as base64
def encode_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# App layout
def app():
    # Background image using custom CSS
    background_image_style = f"""
        background-image: url("data:image/jpeg;base64,{encode_image_as_base64("C:/Users/HomePC/Documents/weed_detect/mini_project/white.jpg")}");  
        background-size: cover;
        padding: 20px;
    """

    st.markdown(
        f"""
        <style>
            .stApp {{
                {background_image_style}
            }}
            .title {{
                font-size: 40px;
                color: #2E8B57;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 20px;
            }}
            .result {{
                font-size: 30px;
                color: #FF4500;
                text-align: center;
                margin-top: 20px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="title">🌾 Crop Recommendation App</div>', unsafe_allow_html=True)

    # Sidebar instructions
    st.sidebar.subheader("🌱 Nutrient Levels")
    st.sidebar.markdown('- **Nitrogen**: Enter the Nitrogen content.')
    st.sidebar.markdown('- **Phosphorus**: Enter the Phosphorus content.')
    st.sidebar.markdown('- **Calcium**: Enter the Calcium content.')
    st.sidebar.markdown('- **Temperature**: Enter the Temperature amount.')
    st.sidebar.markdown('- **Humidity**: Enter the Humidity amount.')
    st.sidebar.markdown('- **pH**: Enter the PH level.')
    st.sidebar.markdown('- **Rainfall**: Enter the Rainfall Amount.')

    # Sliders for user input
    col1, col2 = st.columns(2)
    with col1:
        Nitrogen = st.slider("Nitrogen", 0.0, 140.0, 50.0, 0.1)
        Phosphorus = st.slider("Phosphorus", 5.0, 145.0, 54.0, 0.1)
        Calcium = st.slider("Calcium", 5.0, 205.0, 48.0, 0.1)
        temperature = st.slider("Temperature", 8.0, 44.0, 25.6, 0.1)

    with col2:
        humidity = st.slider("Humidity", 14.0, 100.0, 71.4, 0.1)
        ph = st.slider("pH", 3.0, 10.0, 6.0, 0.1)
        rainfall = st.slider("Rainfall", 20.2, 299.0, 103.0, 0.1)

    # Button to trigger prediction
    if st.button('🚀 Predict Crop Recommendation', key='prediction_button', help="Click to predict crop recommendation"):
        with st.spinner('🔄 Predicting...'):
            # Make prediction using the provided function
            prediction_result = make_prediction(Nitrogen, Phosphorus, Calcium, temperature, humidity, ph, rainfall)

            # Display prediction result
            st.markdown(f'<div class="result">🌾 The recommended crop is: <b>{prediction_result}</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    app()
