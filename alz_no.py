import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


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
    # 구분 하는 모델
    model = load_model(r'C:\pjbm\xray8\alz_bc\model\tf_model_1.h5')

    # 업로드된 이미지를 선택
    file = st.sidebar.file_uploader("Choose a Brain image file", type=['jpeg', 'jpg', 'png'])

    if file is not None:
        image = Image.open(file)
        
        # 이미지를 RGB로 변환
        image = image.convert('RGB')
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        st.image(image, use_column_width=True)
        max_probability = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        class_labels = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        predicted_class_label = class_labels[predicted_class_index]
        st.write(f"Predicted Class: {predicted_class_label}")
        st.write(f"Probability: {max_probability:.4f}")


        if st.button('gradcam'):           
            uploaded_image =r'C:\pjbm\xray8\gradcam_result.jpg'
            images = Image.open(uploaded_image)
            st.image(images)

if __name__ == "__main__":
    app()
