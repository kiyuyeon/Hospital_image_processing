import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

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


def app():
    st.title("Breast Cancer Oncology")

    # Sidebar with file uploader
    uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert the uploaded file to a PIL Image
        up_image = Image.open(uploaded_file).resize((128, 128))
        im_array = np.array(up_image)

        # Show the original image in the sidebar
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(im_array, cmap="gray")
        st.sidebar.pyplot(fig)

        # Prepare image for model prediction
        img_array = keras.utils.img_to_array(up_image)
        img_array = np.expand_dims(img_array, axis=0)

        # Load the pre-trained model
        model = load_model(r'C:\pjbm\xray7\my_cnn_model.h5')
        last_conv_layer_name = "conv2d_4"
        model.layers[-1].activation = None

        # Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        # 2. Convert heatmap to RGB format
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((im_array.shape[1], im_array.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        cam_image = save_and_display_gradcam(up_image, heatmap)

        # Normalize the image and predict using the model
        img_array /= 255.0
        predictions = model.predict(img_array)

        # Interpret the prediction results
        if predictions <= 0.0:
            result = "normal"
            label = "normal"
        else:
            result = "Cancer"
            label = "Cancer"



        # Create a new figure for GradCAM visualization
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Bar chart for predictions
        labels = ["normal", "Cancer"]
        axs[1].barh(labels, [1-predictions[0][0], predictions[0][0]])
        axs[1].set_title('Prediction Probabilities')
        axs[1].set_xlim([0, 1])

        # 3. Display the heatmap over the original image
        # axs[0].imshow(im_array, cmap='bone', zorder=1)
        axs[0].imshow(cam_image, zorder=2)
        axs[0].set_title(f"{label}: {predictions[0][0]:.3f}")
        axs[0].axis('off')

        st.pyplot(fig)