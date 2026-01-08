import numpy as np
import tensorflow as tf


def preprocess_mobilenet_v2(image, target_size=(224, 224)):
    image = image.resize(target_size)
    array = tf.keras.preprocessing.image.img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return array


def load_image_for_mobilenet(path, target_size=(224, 224)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=target_size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.mobilenet.preprocess_input(array)
    return array
