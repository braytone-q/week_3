import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ===== Load the trained MNIST model =====
model = load_model("mnist_model.h5")  # replace with your model path

st.title("MNIST Digit Classifier")
st.write("Draw a digit (0-9) below or upload an image, and the model will predict it!")

# ===== Upload or Draw Input =====
choice = st.radio("Choose input method:", ["Upload Image", "Draw Digit"])

if choice == "Upload Image":
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
elif choice == "Draw Digit":
    st.write("Draw a digit on the canvas below")
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

# ===== Preprocess & Predict =====
if 'image' in locals():
    # Resize to 28x28
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image)/255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Show the input image
    st.image(image, caption="Input Image", width=150)

    # Predict
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"Predicted Digit: **{predicted_digit}**")
    st.write(f"Confidence: **{confidence:.2f}**")
