import streamlit as st
import tensorflow as tf
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import keras


# 모델 로드하기
def load_model():
    return keras.models.load_model(r"C:\pjbm\xray6\ab_model\model4ATD.h5")


model = load_model()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
def visualize_dicom_image(dicom_data):
    fig, ax = plt.subplots()  # 여기서 fig와 ax를 함께 생성합니다
    image = dicom_data.pixel_array
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)  # 수정: figure 객체를 명시적으로 전달합니다

def map_predictions_to_labels(predictions):
    mapped = {}
    
    # Bowel
    if predictions[0][0] > 0.5:
        mapped["Bowel"] = "Injury"
    else:
        mapped["Bowel"] = "Healthy"

    # Extravasation
    if predictions[1][0] > 0.5:
        mapped["Extravasation"] = "Injury"
    else:
        mapped["Extravasation"] = "Healthy"

    # Kidney
    kidney_labels = ["Healthy", "Low Injury", "High Injury"]
    mapped["Kidney"] = kidney_labels[np.argmax(predictions[2])]

    # Liver
    liver_labels = ["Healthy", "Low Injury", "High Injury"]
    mapped["Liver"] = liver_labels[np.argmax(predictions[3])]

    # Spleen
    spleen_labels = ["Healthy", "Low Injury", "High Injury"]
    mapped["Spleen"] = spleen_labels[np.argmax(predictions[4])]

    return mapped

def dcm_to_jpg(ds):

    # DICOM 이미지 데이터를 8 비트로 스케일링
    image_data = ds.pixel_array
    image_data_scaled = ((image_data.astype(np.float32) / image_data.max()) * 255).astype(np.uint8)

    # 8 비트 이미지를 Pillow 이미지로 변환
    image = Image.fromarray(image_data_scaled)

    # JPG로 저장
    jpg_file_path = 'output.jpg'
    image.save(jpg_file_path)

def app():
    st.title("Abdominal Trauma Detection")

    # DICOM 파일을 업로드하기 위한 위젯
    uploaded_file = st.sidebar.file_uploader("Choose a DICOM image...", type=["dcm", "dicom"])
    
    if uploaded_file:
        # pydicom으로 DICOM 파일 로드
        dicom_data = pydicom.dcmread(uploaded_file)
        dcm_to_jpg(dicom_data)

        # DICOM 이미지 시각화
        st.subheader("DICOM Image")
        col1, col2 = st.columns(2)
        with col1:
            visualize_dicom_image(dicom_data)
        
    
        # 예측 시작 버튼
        with col2:
            if st.button("Predict"):
                # 모델 예측 (적절한 전처리가 필요합니다)
                image = dicom_data.pixel_array
                image_3d = np.expand_dims(image, axis=-1)  # 3차원으로 확장
                image_resized = tf.image.resize(image_3d, [256, 256])
                image_rgb = tf.stack([image_resized[:,:,0], image_resized[:,:,0], image_resized[:,:,0]], axis=-1).numpy()

                image_processed = np.expand_dims(image_rgb, axis=0)  # 모델에 맞게 차원을 조절합니다
                prediction = model.predict(image_processed)
                label = ["Bowel", "Extravasation", "Kidney", "Liver", "Spleen"]

                mapped_labels = map_predictions_to_labels(prediction)

                # 매핑된 레이블과 값을 Streamlit을 사용하여 출력합니다.
                for label, value in mapped_labels.items():
                    st.write(f"{label}: {value}")
                # # 예측 결과 출력 (이 부분은 사용하는 모델과 작업에 따라 조절할 필요가 있습니다)
                # st.subheader("Prediction Results")
                # st.write(prediction)
        
        if st.button('Show Masks'):

            image_sam =cv2.imread("./output.jpg")
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            image_array = np.asarray(image_sam)
            sam_checkpoint = r"C:\pjbm\xray6\tumor\sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            device = "cpu"

            
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(image_array)  # PIL.Image 객체가 아니라 numpy array 형태의 이미지 사용

            plt.figure(figsize=(10,10))
            plt.imshow(image_array)  # PIL.Image 객체가 아니라 numpy array 형태의 이미지 사용
            show_anns(masks)
            plt.axis('off')

            # Figure 객체를 Streamlit에서 보여줍니다.
            st.pyplot(plt.gcf())
