import streamlit as st
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import math
from collections import OrderedDict
from Prostate.util import CustomSEResNeXt, get_transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()
    
def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img

def plotGradCAM(model, final_conv, fc_params, train_loader, 
                row=1, col=8, img_size=256, device='cpu', original=False):
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    # save activated_features from conv
    activated_features = SaveFeatures(final_conv)

    # save weight from fc
    weight = np.squeeze(fc_params[0].cpu().data.numpy())

    # original images
    if original:
        st.subheader("Original Images")
        cols = st.columns(col)
        for i, img in enumerate(train_loader):
            output = model(img.to(device))
            pred_idx = output.to('cpu').numpy().argmax(1)
            cur_images = img.numpy().transpose((0, 2, 3, 1))
            cols[i % col].image(cur_images[0], use_column_width=True, caption=f'Label:{target}, Predict:{pred_idx}')
            if i == row * col - 1:
                break

    # heatmap images
    st.subheader("Heatmap Images")
    cols = st.columns(col)
    for i, (img, target, _) in enumerate(train_loader):
        output = model(img.to(device))
        pred_idx = output.to('cpu').numpy().argmax(1)
        cur_images = img.cpu().numpy().transpose((0, 2, 3, 1))
        heatmap = getCAM(activated_features.features, weight, pred_idx)

        overlay = cv2.cvtColor(cur_images[0], cv2.COLOR_BGR2RGB)
        overlay = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # 이렇게 overlay 이미지를 생성할 수 있습니다. alpha값을 조절하여 필요한 만큼의 투명도를 설정하세요.
        overlayed_img = cv2.addWeighted(overlay, 0.4, cur_images[0], 1 - 0.4, 0)

        cols[i % col].image(overlayed_img, use_column_width=True, caption=f'Label:{target}, Predict:{pred_idx}')
        if i == row * col - 1:
            break

class CustomDataset(Dataset):
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor

    def __len__(self):
        return 1  # 단일 이미지만 있습니다.

    def __getitem__(self, idx):
        return self.image_tensor 


def app():
    # 모델 정의 및 가중치 로딩
    model = CustomSEResNeXt()
    model.load_state_dict(torch.load(r"C:\Users\bfff\pjbm\xray7\Prostate\custom_se_resnext50_32x4d_updated.pth"))
    model.eval()

    st.title("Grad-CAM Visualization with Streamlit")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("Upload a file", type=["tiff"])

    
    if uploaded_file is not None:
        try:
            # TIFF 이미지 로드
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        except Exception as e:
            st.write(f"Error opening the TIFF image: {e}")
        
        image_np = np.array(image)

        # 이미지 변환
        transform = get_transforms(data='valid')  # 'valid' 변환 적용
        transformed_image = transform(image=image_np)  # 변환 적용
        image_tensor = transformed_image['image'].unsqueeze(0)  # Add a batch dimension
        
        # 모델을 사용하여 예측
        with torch.no_grad():
            prediction = model(image_tensor)

        # Display the prediction (modify as per your needs)
        st.write(f"Prediction: {prediction}")

if __name__ == "__main__":
    app()
    
    
#         image_tensor = transformed_image['image']
        
#         # 모델의 필요한 부분 가져오기
#         final_conv = model.model.layer4[2]._modules.get('conv3')
#         fc_params = list(model.model._modules.get('last_linear').parameters())
        

#         dataset = CustomDataset(image_tensor)  # 이미지 텐서를 사용하여 CustomDataset 정의 필요
#         train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
#         # 모델의 필요한 부분 가져오기
#         final_conv = model.model.layer4[2]._modules.get('conv3')
#         fc_params = list(model.model._modules.get('last_linear').parameters())
        
#         # 모델을 사용하여 예측 및 Grad-CAM 생성
#         plotGradCAM(model, final_conv, fc_params, train_loader, img_size=256, device='cpu', original=True)

# if __name__ == "__main__":
#     app()
            
        
