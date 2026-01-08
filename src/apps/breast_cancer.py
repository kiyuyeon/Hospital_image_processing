from _bootstrap import add_repo_root

add_repo_root()

import streamlit as st
import numpy as np
from tensorflow import keras
from keras.models import load_model

from src.utils.keras_gradcam import apply_heatmap, make_gradcam_heatmap

# Main Streamlit app
st.title("Grad-CAM Class Activation Visualization")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Save uploaded image to a temporary file
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Load the image from the temporary file
    image = keras.utils.load_img("temp_image.jpg", target_size=(128, 128))
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Prepare image
    img_array = keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Make model
    model_builder = load_model(r'E:\data\my_cnn_model.h5')
    model = model_builder
    last_conv_layer_name = "conv2d_4"
    model.layers[-1].activation = None

    # Get Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Save and display Grad-CAM
    cam_image = apply_heatmap(image, heatmap)
    st.image(cam_image, caption="Grad-CAM Heatmap.", use_column_width=True)
    
    #결과값 예상
    img_array /= 255.0 #정규화
    # 이미지를 모델로 예측
    predictions = model.predict(img_array)
    # 예측 결과 해석 및 출력
    if predictions <= 0.0:
        st.write(predictions)
        result = "정상"
    else:
        st.write(predictions)
        result = "암"

    st.write("예측 결과:", result)
