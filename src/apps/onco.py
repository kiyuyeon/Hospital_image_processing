import streamlit as st

import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
import pickle

import sklearn

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

from PIL import Image


def app():
  st.title("Colorectal Histology Classification")
  st.set_option('deprecation.showfileUploaderEncoding', False)

  #upload a slide image of the colorectal histology
  uploaded_file = st.sidebar.file_uploader("Choose a colorectal tissue slide image file")

  def load_preprocess(upload_file):
    if upload_file is not None:
      im1 = Image.open(upload_file)
      im1 = tf.keras.preprocessing.image.img_to_array(im1)
      im1 = tf.cast(im1, tf.float32) / 255.
      im1_tensor = np.expand_dims(im1,axis=0)
      
      fig1 = plt.figure()
      plt.axis('off')
      plt.imshow(im1)
      st.sidebar.pyplot(fig1)

    return im1, im1_tensor


  model = tf.keras.models.load_model('model4.h5')
  #model.summary()

  baseline_im = tf.ones(shape=(150,150,3))

  def interpolate_images(baseline,
                        image,
                        alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images

  #@st.cache 
  def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
      tape.watch(images)
      logits = model(images)
      probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


  def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

    
  #@tf.function
  def integrated_gradients(baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
      from_ = alpha
      to = tf.minimum(from_ + batch_size, len(alphas))
      alpha_batch = alphas[from_:to]

      # 2. Generate interpolated inputs between baseline and input.
      interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                        image=image,
                                                        alphas=alpha_batch)

      # 3. Compute gradients between model outputs and interpolated inputs.
      gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                        target_class_idx=target_class_idx)

      # Write batch indices and gradients to extend TensorArray.
      gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

    
  def plot_img_attributions(baseline,
                            image,
                            target_class_idx,
                            m_steps=50,
                            cmap=None,
                            overlay_alpha=0.4):

    attributions = integrated_gradients(baseline=baseline,
                                        image=image,
                                        target_class_idx=target_class_idx,
                                        m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Attribution mask')
    axs[0, 0].imshow(attribution_mask, cmap=cmap)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Overlay')
    axs[0, 1].imshow(attribution_mask, cmap=cmap)
    axs[0, 1].imshow(image, alpha=overlay_alpha)
    axs[0, 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)
    return fig

  lab_list = ['Tumor', 'Stroma','Complex', 'Lympho','Debris','Mucosa','Adipose','Empty']

  # Get the user-uploaded image and classify it
  if uploaded_file is not None:

    im1 = Image.open(uploaded_file)
    im1 = tf.keras.preprocessing.image.img_to_array(im1)
    im1 = tf.cast(im1, tf.float32) / 255.
    im1_tensor = np.expand_dims(im1,axis=0)
      
    #fig1 = plt.figure()
    fig1, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(12, 6))
    
    axs[0, 0].set_title('Original image')
    axs[0, 0].imshow(im1)
    axs[0, 0].axis('off')

    predictionz = model.predict(im1_tensor)
    sorted_preds, sorted_labels = (list(reversed(t)) for t in zip(*sorted(zip(predictionz[0], lab_list))))
    axs[0, 1].set_title('Predictions')
    axs[0, 1].barh(sorted_labels, sorted_preds)
    st.pyplot(fig1)
    
    ig_attributions = integrated_gradients(baseline=baseline_im, image=im1, target_class_idx=0, m_steps=480)
                                        
    _ = plot_img_attributions(image=im1, baseline=baseline_im, target_class_idx=0, m_steps=480, cmap=plt.cm.hot, overlay_alpha=0.4)

# 예시 이미지 넣는 코드
# else:
#   im1 = Image.open('content/01Tumor.tif')
#   im1 = tf.keras.preprocessing.image.img_to_array(im1)
#   im1 = tf.cast(im1, tf.float32) / 255.
#   im1_tensor = np.expand_dims(im1,axis=0)
    
#   #fig1 = plt.figure()
#   fig1, axs = plt.subplots(nrows=2, ncols=1, squeeze=False, figsize=(6, 6))
  
#   axs[0, 0].set_title('Default image')
#   axs[0, 0].imshow(im1)
#   axs[0, 0].axis('off')

  
#   predictionz = model.predict(im1_tensor)
#   sorted_preds, sorted_labels = (list(reversed(t)) for t in zip(*sorted(zip(predictionz[0], lab_list))))
#   axs[1, 0].set_title('Predictions')
#   axs[1, 0].barh(sorted_labels, sorted_preds)

#   st.pyplot(fig1)
  
#   ig_attributions = integrated_gradients(baseline=baseline_im, image=im1, target_class_idx=0, m_steps=480)
                                       
#   _ = plot_img_attributions(image=im1, baseline=baseline_im, target_class_idx=0, m_steps=480, cmap=plt.cm.hot, overlay_alpha=0.4)
  


