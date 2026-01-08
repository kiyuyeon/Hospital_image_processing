import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.inference.predict import predict_single_image_with_gradcam  # noqa: E402
from src.models.vgg16_classifier import VGG16Classifier  # noqa: E402


st.set_page_config(page_title="Grad-CAM Demo", layout="wide")
st.title("VGG16 Grad-CAM Demo")

with st.sidebar:
    st.header("Model")
    model_path = st.text_input(
        "Model path",
        value=str(ROOT / "artifacts" / "model_VGG16.pt"),
    )
    labels_text = st.text_input(
        "Class labels (comma-separated)",
        value="MildDemented,ModerateDemented,NonDemented,VeryMildDemented",
    )
    labels = [x.strip() for x in labels_text.split(",") if x.strip()]
    num_classes = st.number_input(
        "Number of classes",
        min_value=2,
        max_value=100,
        value=max(2, len(labels)),
        step=1,
    )
    device_choice = st.selectbox("Device", ["auto", "cpu"])

device = "cuda" if torch.cuda.is_available() and device_choice == "auto" else "cpu"


@st.cache_resource
def load_model(path, classes, device_name):
    model = VGG16Classifier(num_classes=classes).to(device_name)
    state = torch.load(path, map_location=device_name)
    model.load_state_dict(state)
    return model


st.header("Input")
source = st.radio("Image source", ["Upload", "Local path"])
image = None

if source == "Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
else:
    local_path = st.text_input("Local image path", value="")
    if local_path:
        image = Image.open(local_path)

run = st.button("Run Grad-CAM", type="primary")

if run:
    if not model_path or not Path(model_path).exists():
        st.error("Model path is invalid.")
        st.stop()
    if image is None:
        st.error("Please provide an image.")
        st.stop()

    model = load_model(model_path, int(num_classes), device)
    result = predict_single_image_with_gradcam(
        model=model,
        image=image,
        class_labels=labels,
        device=device,
    )

    st.subheader("Result")
    st.write(
        f"Prediction: {result['label']} | Confidence: {result['confidence']:.2f}%"
    )

    col1, col2, col3 = st.columns(3)
    col1.image(result["input_image"], caption="Input", use_container_width=True)
    col2.image(result["heatmap"], caption="Grad-CAM", use_container_width=True)
    col3.image(result["overlay"], caption="Overlay", use_container_width=True)
