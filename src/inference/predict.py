from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.visualization.gradcam import GradCAM


def _load_image(image):
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def predict_single_image_with_gradcam(
    model,
    image,
    class_labels,
    device,
    target_size=(224, 224),
):
    tf = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    img = _load_image(image)
    x = tf(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100

    cam_extractor = GradCAM(model)
    cam = cam_extractor.generate(x, pred_idx)[0].detach().cpu().numpy()

    cam = np.uint8(255 * cam)
    cam = cv2.resize(cam, target_size)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_np = x[0].cpu().permute(1, 2, 0)
    img_np = (
        img_np * torch.tensor([0.229, 0.224, 0.225])
        + torch.tensor([0.485, 0.456, 0.406])
    )
    img_np = img_np.clamp(0, 1).detach().numpy()

    overlay = 0.5 * (heatmap / 255.0) + img_np
    overlay = np.clip(overlay, 0, 1)

    label = (
        class_labels[pred_idx]
        if class_labels and pred_idx < len(class_labels)
        else str(pred_idx)
    )

    return {
        "pred_idx": pred_idx,
        "label": label,
        "confidence": confidence,
        "input_image": img_np,
        "heatmap": heatmap,
        "overlay": overlay,
    }
