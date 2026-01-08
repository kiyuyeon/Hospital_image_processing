from _bootstrap import add_repo_root

add_repo_root()

import streamlit as st
import numpy as np
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.keras_gradcam import apply_heatmap, make_gradcam_heatmap

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
        cam_image = apply_heatmap(up_image, heatmap)

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
