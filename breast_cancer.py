import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image

# Function to get image array
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Function to create Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display Grad-CAM
def save_and_display_gradcam(img_array, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.img_to_array(img_array)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return superimposed_img

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
    cam_image = save_and_display_gradcam(image, heatmap)
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