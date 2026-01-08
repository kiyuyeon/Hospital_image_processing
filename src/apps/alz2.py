import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

def compute_gradcam(model, img_array, layer_name):
    # 입력 이미지 전처리
    img = tf.keras.preprocessing.image.load_img(img_array, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet.preprocess_input(img)  # 모델에 맞는 전처리 함수 사용

    # 모델의 예측 클래스 결정
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])

    # GradCAM 모델 생성
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    # Gradient를 계산하여 가중치화된 특성 맵 얻기
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 특성 맵과 가중치화된 Gradient를 곱하여 GradCAM 생성
    heatmap = tf.reduce_mean(tf.multiply(conv_output[0], pooled_grads), axis=-1)
    heatmap = tf.keras.backend.eval(tf.maximum(heatmap, 0) / tf.reduce_max(heatmap))
    
    return heatmap

# 구분 모델 이미지 전처리
def preprocess_image(image, target_size=(224, 224)):
    # 이미지 크기 조정 및 전처리
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image


def app():
    st.title("Alzheimer's Image Classification")
    st.text("Detect the presence of Alzheimer's disease.")
    
    # Load the model
    model_path = r'C:\pjbm\xray8\alz_bc\model\tf_model_1.h5'
    model = load_model(model_path)

    # Upload the image
    upload_file = st.sidebar.file_uploader("Choose a Brain image file", type=['jpeg', 'jpg', 'png'])


    
    if upload_file is not None:
        up_image = Image.open(upload_file)   
        
        # Convert the image to numpy array
        im_array = np.array(up_image)
        
        # Show the original image in the sidebar
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(im_array, cmap="gray")
        
        st.sidebar.pyplot(fig)
        
        # Convert and display uploaded image
        image = Image.open(upload_file).convert('RGB')
        
        # Save the uploaded file temporarily
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(upload_file.getvalue())
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        
        
        class_labels = ["Mild", "VeryMild", "Moderate", "Non"]
        
        
        # max_probability = np.max(predictions)
        # predicted_class_label = class_labels[np.argmax(predictions)]
        # st.write(f"Predicted Class: {predicted_class_label}")
        # st.write(f"Probability: {max_probability:.4f}")

        # Compute GradCAM
        layer_name = 'Conv_1'
        heatmap = compute_gradcam(model, temp_file_path, layer_name)
        img = cv2.imread(temp_file_path)
        img = cv2.resize(img, (224, 224))
        
        # Apply heatmap on original image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save the result
        save_path = "gradcam_result.jpg"
        cv2.imwrite(save_path, superimposed_img)

        # # If the button is pressed, display the GradCAM result
        # images = Image.open(save_path)
        # st.image(images)

#--------------------------------------------------------------------------------
        # GradCAM processing
        sorted_preds, sorted_labels = (list(reversed(t)) for t in zip(*sorted(zip(predictions[0], class_labels))))

        # Create a new figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Bar chart for predictions
        axs[1].barh(sorted_labels, sorted_preds)
        axs[1].set_title('Prediction Demented Probabilities')
        axs[1].set_xlim([0, 1])

        # GradCAM visualization
        i = 0
        axs[0].imshow(img, cmap='bone')
        axs[0].imshow(superimposed_img, cmap='gray', alpha=min(0.5, sorted_preds[i]))
        axs[0].set_title(sorted_labels[i] + ": " + str(round(sorted_preds[i], 3)))
        axs[0].axis('off')

        st.pyplot(fig)
        
if __name__ == "__main__":
    app()