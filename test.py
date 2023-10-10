import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 모델 경로 지정
model_path = r'D:\alzheimer\models\tf_model_1.h5'

# 모델 불러오기
model = tf.keras.models.load_model(model_path)

def compute_gradcam(model, img_array, layer_name):
    # 입력 이미지 전처리
    img = tf.keras.preprocessing.image.load_img(img_array, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet.preprocess_input(img)  # 모델에 맞는 전처리 함수 사용

    # 모델의 예측 클래스 결정
    preds = model.predict(img)
    class_idx = np.argmax(preds[0])

    # GradCAM 모델 생성
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

    # Gradient를 계산하여 가중치화된 특성 맵 얻기
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 특성 맵과 가중치화된 Gradient를 곱하여 GradCAM 생성
    heatmap = tf.reduce_mean(tf.multiply(conv_output[0], pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap=np.array(heatmap)
    return heatmap

# GradCAM을 적용할 이미지 경로
image_path = r'C:\pjbm\xray8\gra.jpg'


# GradCAM을 계산할 레이어 이름 (예: 'conv2d_34')
layer_name = 'out_relu'

# GradCAM 계산
heatmap = compute_gradcam(model, image_path, layer_name)

# 원본 이미지 로드
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))

# heatmap을 원본 이미지 크기로 확대
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# heatmap을 원본 이미지에 적용
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 원본 이미지와 heatmap을 결합하여 시각화
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# 결과 시각화
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

save_path = "gradcam_result.jpg"  # 저장할 경로와 파일 이름을 설정하세요

# GradCAM 결과 이미지 저장
cv2.imwrite(save_path, superimposed_img)

# 이미지 저장이 완료되면 메시지를 출력합니다.
print(f"GradCAM 결과 이미지가 {save_path}에 저장되었습니다.")