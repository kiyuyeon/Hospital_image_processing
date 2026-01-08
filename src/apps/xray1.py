import streamlit as st
from pathlib import Path
from PIL import Image
from keras.models import load_model
from neural_network_model.transfer_learning import TransferModel

# upload file
file = st.sidebar.file_uploader('', type=['jpeg', 'jpg', 'png'])

# 모델 로드
model_path = "C:\\Users\\bfff\\pjbm\\xray1\\models\\tf_model_1.h5"
model = load_model(model_path)


def main():
    st.title("DrillBitVision Neural Network Model Analysis")

    # TransferModel 인스턴스 생성
    current_path = Path(__file__).parent
    dataset_address = current_path / "dataset"
    transfer_model = TransferModel(dataset_address=dataset_address)

    # 모델 로드
    model_path = "C:\\Users\\bfff\\pjbm\\xray1\\models\\tf_model_1.h5"
    
    # 테스트 예측
    if st.button("Predict Test"):
        report = transfer_model.predict_test(model_path=model_path)
        st.write(report)

    # Grad-CAM 시각화
    st.subheader("Grad-CAM Visualization")
    transfer_model.grad_cam_viz(num_rows=3, num_cols=2)
    st.pyplot()

if __name__ == "__main__":
    main()
