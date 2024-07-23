import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Load the trained model

model = load_model('deepfake_detection_model.h5')

# Preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Predict if the image is fake or real
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"

# Streamlit application
st.markdown("<h1 style='text-align: center; color: grey;'>DEEP FAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>", unsafe_allow_html=True)
st.image("coverpage.png")

# Detailed description about deepfake
st.header("Understanding Deepfakes")
st.write("""
Deepfakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness. Leveraging sophisticated AI algorithms, primarily deep learning techniques, deepfakes can create incredibly realistic and convincing fake videos and images. This technology, while having legitimate uses in entertainment and education, poses significant ethical and security challenges. Deepfakes can be used to spread misinformation, create malicious content, and impersonate individuals without consent, raising serious concerns about privacy and trust in digital media. Detection of deepfakes is crucial to mitigate these risks, and AI plays a vital role in identifying such manipulations. By analyzing subtle artifacts and inconsistencies that are often imperceptible to the human eye, AI models can effectively distinguish between real and fake media, ensuring the integrity of visual content.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # To read file as bytes:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR")

    # Predict and display result
    result = predict_image(image)

    # Set the color based on the result
    # Set the color based on the result
    if result == "Fake":
        color = "red"
        description = "Our deepfake detection model has classified this image as fake based on various factors. Deepfake images often exhibit certain artifacts or inconsistencies that are not present in real images. These could include mismatched facial features, unnatural lighting or shadows, or inconsistencies in facial expressions. Our model has been trained to recognize these patterns and distinguish between real and fake images with high accuracy."

    elif result == "Real":
        color = "green"
        description = "Our deepfake detection model has classified this image as real. Real images typically lack the subtle anomalies and inconsistencies present in deepfake images. Our model has been trained on a diverse dataset of real and fake images, enabling it to accurately differentiate between the two categories."

    # Display the title with the appropriate color
    st.markdown(f"<h1 style='color:{color};'>The image is {result}</h1>", unsafe_allow_html=True)

    # Display the description
    st.write(description)


st.title("Model Training Graph")
st.markdown("### Model Training accuracy: 95%")
st.image("Figure_2.png")
st.markdown("### Model Training Loss")
st.image("Figure_1.png")

# Footer section
st.markdown("""
---
**Contact Us:**
For more information and queries, please contact us at [contact@example.com](mailto:contact@example.com).

**Follow us on:**
[Twitter](https://twitter.com) | [LinkedIn](https://linkedin.com) | [Facebook](https://facebook.com)
""")
