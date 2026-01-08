import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from tumor.util import visualize

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=True):
    if len(mask.shape) == 2:
        # Handle the case where mask has only 2 dimensions
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([1, 0, 0, 0.3])
        h, w = mask.shape
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    elif len(mask.shape) == 3:
        # Handle the case where mask has 3 dimensions
        num_masks = mask.shape[0]
        for m in range(num_masks):
            single_mask = mask[m]
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([1, 0, 0, 0.3])
            h, w = single_mask.shape
            mask_image = single_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=True,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


def app():
    # Streamlit 앱 제목 설정
    st.title('Brain MRI Tumor detection')

    # 파일 업로더를 통해 이미지 업로드
    file = st.sidebar.file_uploader('', type=['png', 'jpg', 'jpeg'])
    
    #load model and image
    MedSAM_CKPT_PATH = r"C:\pjbm\xray6\MedSAM\medsam_vit_b.pth"
    ''
    # Load the state dictionary manually, ensuring it's loaded to the CPU
    checkpoint = torch.load(MedSAM_CKPT_PATH, map_location=torch.device('cpu'))
    device = "cuda"

    # Initialize the model without loading the weights (assuming this is possible with the function call)
    medsam_model = sam_model_registry['vit_b']()

    # Manually update the model's weights
    medsam_model.load_state_dict(checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()


    if file:
        img_np = io.imread(file)
        
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
            
        H, W, _ = img_3c.shape

        # image preprocessing and model inference
        img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # 모델 설정
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = r'C:\pjbm\xray6\tumor\model\model.pth'
    cfg.MODEL.DEVICE = 'cuda'

    # 모델 예측기 생성
    predictor = DefaultPredictor(cfg)

    # 이미지 로드 및 처리
    if file:
        image = Image.open(file).convert('RGB')

        image_array = np.asarray(image)

        # 객체 감지 수행
        outputs = predictor(image_array)

        threshold = 0.5

        # 예측 결과 가져오기
        preds = outputs["instances"].pred_classes.tolist()
        scores = outputs["instances"].scores.tolist()
        bboxes = outputs["instances"].pred_boxes

        bboxes_ = []
        for j, bbox in enumerate(bboxes):
            bbox = bbox.tolist()

            score = scores[j]
            pred = preds[j]

            if score > threshold:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                bboxes_.append([x1, y1, x2, y2])
        
        col1, col2 = st.columns(2)
        with col1:
            # 시각화
            visualize(image, bboxes_)
        
        box_np_list = []  # 빈 리스트로 초기화

        for j, bbox in enumerate(bboxes):
            bbox = bbox.tolist()

            score = scores[j]
            pred = preds[j]

            if score > threshold:
                x1, y1, x2, y2 = [int(i) for i in bbox]
                box_np_list.append([x1-10, y1-10, x2+10, y2+10])  # box를 리스트에 추가

        box_np = np.array(box_np_list)  # numpy 배열로 변환

        # transfer box_np t0 1024x1024 scale
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

        medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))  # only one plot

        ax.imshow(img_3c)
        show_mask(medsam_seg, ax, random_color=False)
        # show_box(box_np[1], ax)
        for box in box_np:
            show_box(box, ax)
        ax.set_title("MedSAM Segmentation")
        
        with col2:
            with st.spinner('Rendering the plot...'):
                st.pyplot(fig)
