import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

# Load the ONNX model
def load_model(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

# Preprocess the image for the model
# List of class names
class_names = ["Cyst", "Normal", "Stone", "Tumor"]  # Update with your actual class names

def preprocess_image(image, target_size=(224, 224)):
    image = np.array(image)
    if len(image.shape) == 2:  # Grayscale image
        image = np.stack((image,) * 3, axis=-1)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    if image.shape[-1] == 3:
        image = np.transpose(image, (0, 3, 1, 2))
    return image

def predict(session, input_name, image):
    pred_onx = session.run(None, {input_name: image})[0]
    probabilities = np.exp(pred_onx) / np.sum(np.exp(pred_onx), axis=1)
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence_score = probabilities[0][predicted_class_idx] * 100
    return predicted_class, confidence_score

def main():
    # Custom page configurations
    st.set_page_config(page_title="ONNX Model Classifier", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    
    # Header section with image
    st.markdown("<h1 style='text-align: center; color: green;'>Image Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: blue;'>Upload an image to get the predicted class and confidence score</h3>", unsafe_allow_html=True)
    
    # Add a sidebar with information or controls
    st.sidebar.markdown("## About the App")
    st.sidebar.write("This app uses an ONNX model to classify images and display the confidence score.")
    st.sidebar.markdown("#### How to Use:")
    st.sidebar.write("1. Upload an image.\n2. Wait for the prediction.\n3. See the results below.")
    
    # Load the ONNX model
    model_path = r"best.onnx"  # Update with your model's path
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            image = Image.open(uploaded_file)

            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            predicted_class, confidence_score = predict(session, input_name, processed_image)

            # Display prediction results in a creative way
            st.success(f"**Predicted Class:** {predicted_class}")
            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>Confidence Score: {confidence_score:.2f}%</h2>", unsafe_allow_html=True)

            # Add a progress bar to visualize confidence
            st.progress(int(confidence_score))
    else:
        st.warning("Please upload an image to start the prediction.")

    # Footer
    st.markdown("<h4 style='text-align: center; color: pink;'>Built by Harshit Mittal under the guidance of Professor Raji Susan Mathew</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
