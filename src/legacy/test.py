from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils.keras_gradcam import make_gradcam_heatmap
from src.utils.tf_image import load_image_for_mobilenet

model_path = r"D:\alzheimer\models\tf_model_1.h5"
model = tf.keras.models.load_model(model_path)

image_path = r"C:\pjbm\xray8\gra.jpg"
layer_name = "out_relu"

img_array = load_image_for_mobilenet(image_path, target_size=(224, 224))
heatmap = make_gradcam_heatmap(img_array, model, layer_name)

img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

save_path = "gradcam_result.jpg"
cv2.imwrite(save_path, superimposed_img)
print(f"GradCAM result saved to {save_path}.")
