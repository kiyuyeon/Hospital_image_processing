import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Xception 모델을 사용할 것입니다.
model_builder = keras.applications.xception.Xception
img_size = (299, 299)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = "block14_sepconv2_act"  # Grad-CAM을 생성할 레이어 이름

def get_img_array(img_path, size):
    # 이미지를 불러와 Numpy 배열로 변환합니다.
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)  # 배치 차원을 추가합니다.
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 마지막 합성곱 레이어의 출력 및 모델의 예측 결과를 얻을 수 있는 모델을 만듭니다.
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 입력 이미지에 대한 예측 결과의 최상위 클래스에 대한 그래디언트를 계산합니다.
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 마지막 합성곱 레이어 출력에 대한 출력 뉴런의 그래디언트입니다.
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 각 피쳐 맵 채널에 대한 그래디언트의 평균을 구합니다.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 각 피쳐 맵 채널에 그래디언트 중요도를 곱하고 모두 더하여 클래스 활성화 맵을 얻습니다.
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 시각화를 위해 활성화 맵을 0과 1 사이로 정규화합니다.
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # 원본 이미지를 로드합니다.
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # 히트맵을 0-255 범위로 조절합니다.
    heatmap = np.uint8(255 * heatmap)

    # 젯 색상 맵을 사용하여 히트맵을 색칠합니다.
    jet = cm.get_cmap("jet")

    # 색칠에 사용할 RGB 값
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # RGB로 색칠된 히트맵 이미지를 만듭니다.
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # 원본 이미지 위에 히트맵을 덧씌웁니다.
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # 합성된 이미지를 저장합니다.
    superimposed_img.save(cam_path)

def image_gra(imag):
    img_path = imag
    # 이미지 준비
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # 모델 생성
    model = model_builder(weights="imagenet")

    # 마지막 소프트맥스 레이어를 제거합니다.
    model.layers[-1].activation = None

    # 최상위 예측 클래스를 출력합니다.
    preds = model.predict(img_array)
    print("예측 결과:", decode_predictions(preds, top=1)[0])

    # Grad-CAM 히트맵 생성
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=285)

    # Grad-CAM 히트맵을 저장하고 표시합니다.
    save_and_display_gradcam(img_path, heatmap)
