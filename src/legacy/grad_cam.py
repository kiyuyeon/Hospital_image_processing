from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt

from src.utils.keras_gradcam import apply_heatmap, get_img_array, make_gradcam_heatmap

model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = "block14_sepconv2_act"

img_path = r"C:\pjbm\xray1\dataset\MildDemented\mildDem1.jpg"

display(Image(img_path))

img_array = preprocess_input(get_img_array(img_path, size=img_size))

model = model_builder(weights="imagenet")
model.layers[-1].activation = None

preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()

cam_image = apply_heatmap(keras.utils.load_img(img_path), heatmap)
cam_image.save("cam.jpg")
display(Image("cam.jpg"))

preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=2)[0])

for pred_index in (260, 285):
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, pred_index=pred_index
    )
    cam_image = apply_heatmap(keras.utils.load_img(img_path), heatmap)
    cam_path = f"cam_{pred_index}.jpg"
    cam_image.save(cam_path)
    display(Image(cam_path))
