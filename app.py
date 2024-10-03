import streamlit as st
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image

# Load the model once and cache it
@st.cache_resource
def load_model_cached():
    model = load_model('Pnemonia.h5')
    return model

model = load_model_cached()

# App title and description
st.title("🫁 Chest X-ray Pneumonia Detection App")
st.write("### Detect Pneumonia from Chest X-ray Images 🌟")
st.write("This application helps detect pneumonia from chest X-ray images using a pre-trained deep learning model. Just upload an image below to get started! 📤")

# Upload image
uploaded_file = st.file_uploader("📥 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Chest X-ray Image', use_column_width=True)
    
    # Image processing
    img = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB format
    img = img.resize((224, 224))  # Resize to 224x224
    img_array = np.array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_data = preprocess_input(img_array)  # Preprocess the image

    st.write("⏳ Analyzing the image...")
    
    # Prediction
    predictions = model.predict(img_data)
    
    # Display results with emojis
    if predictions[0][0] > 0.5:
        st.success("🟢 **Result:** Normal 😊")
    else:
        st.error("⚠️ **Result:** Affected By Pneumonia 😷")

# Footer
st.markdown("---")
st.write("Developed with ❤️ by Amber Bagchi. For any queries, please contact me at amberbagchi.work@gmail.com!")
