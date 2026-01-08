#############################################################################
# Chest X-ray Image Analysis
# Adapted from the AI for Medicine Specialization by deeplearning.ai
# Deployed using Streamlit and Heroku
# Joshua D John for safeXimaging
#############################################################################



import streamlit as st

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import tensorflow.keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
import pickle

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import sklearn
import lifelines
import shap

from PIL import Image

def app():
        
    st.title("Chest X-ray Image Classification")
    st.text("14개의 폐질병을 분류하며 AI가 이 질병이라고 판단을 위한 곳에 Heatmap을 그려줍니다")
    st.text("주의: 이 모델은 다분류 & GradCam을 가능성을 보기 위한 모델로 성능은 다소 떨어집니다.")

    st.set_option('deprecation.showfileUploaderEncoding', False)

    #upload a png image of the chest X-ray
    uploaded_file = st.sidebar.file_uploader("Choose a chest x-ray image file")

    #############################
    # Pre-processing functions
    #############################
    def load_and_preprocess(upload_file, W, H, mean, std):
        """
        Load and preprocess user-uploaded image
        
        Args:
            upload_file -   the png file uploaded by a user
            W, H        -   the resized width and height for the application - 320 x 320 basically
            mean, std   -   mean and standard deviation for normalizing the images
        """
        
        if upload_file is not None:
            up_image = Image.open(upload_file)   
            
        im_array = np.array(up_image)
        
        # Show the original image in the side-bar
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(im_array, cmap="bone")
        st.sidebar.pyplot(fig)
        
        # Resize, normalize, 3-channel and expand the dimensions of the image
        target_size=(H, W)
        x = up_image.resize(target_size)
        x -= mean
        x /= std
        
        y = np.stack((x,x,x), axis=2)
        y = np.expand_dims(y, axis=0)
        
        # x is the normalized image
        # y is the tensor for the model predictor
        return x, y


    def get_mean_std_per_batch(df, H=320, W=320):
        sample_data = []
        for idx, img in enumerate(df.sample(100)["Image"].values):
            path = IMAGE_DIR + img
            sample_data.append(np.array(image.load_img(path, target_size=(H, W))))

        mean = np.mean(sample_data[0])
        std = np.std(sample_data[0])
        return mean, std    

    ######################################
    # Grad CAM functions
    ######################################
    def grad_cam(input_model, image, category_index, layer_name):
        """
        GradCAM method for visualizing input saliency.
        
        Args:
            input_model (Keras.model): model to compute cam for
            image (tensor): input to model, shape (1, H, W, 3)
            cls (int): class to compute cam with respect to
            layer_name (str): relevant layer in model
            H (int): input height
            W (int): input width
        Return:
            cam ()
        """
        cam = []
        

        # Get placeholders for class output and last layer
        # Get the model's output
        output_with_batch_dim = input_model.output
        
        # Remove the batch dimension
        output_all_categories = output_with_batch_dim[0]
        
        # Retrieve only the disease category at the given category index
        y_c = output_all_categories[category_index]
        
        # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
        spatial_map_layer = input_model.get_layer(layer_name).output

        # Get gradients of last layer with respect to output

        # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
        grads_l = K.gradients(y_c, spatial_map_layer)
        
        # Get the gradient at index 0 of the list
        grads = grads_l[0]
            
        # Get hook for the selected layer and its gradient, based on given model's input
    
        spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
        
        # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
        spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

        # Reshape activations and gradient to remove the batch dimension
        # Shape goes from (B, H, W, C) to (H, W, C)
        # B: Batch. H: Height. W: Width. C: Channel    
        # Reshape spatial map output to remove the batch dimension
        spatial_map_val = spatial_map_all_dims[0]
        
        # Reshape gradients to remove the batch dimension
        grads_val = grads_val_all_dims[0]
        
        # Compute weights using global average pooling on gradient 
        # grads_val has shape (Height, Width, Channels) (H,W,C)
        # Take the mean across the height and also width, for each channel
        # Make sure weights have shape (C)
        weights = np.mean(grads_val, axis=(0,1))
        
        # Compute dot product of spatial map values with the weights
        cam = np.dot(spatial_map_val, weights)

        
        H, W = image.shape[1], image.shape[2]
        cam = np.maximum(cam, 0) # ReLU so we only get positive importance
        cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
        cam = cam / cam.max()

        return cam



    def compute_gradcam(model, original_img, img_tensor, mean, std, data_dir, df, 
                        labels, selected_labels, layer_name='conv5_block16_concat'):
        """
        Compute GradCAM for many specified labels for an image. 
        This method will use the `grad_cam` function.
        
        Args:
            model (Keras.model): Model to compute GradCAM for
            img (string): Image name we want to compute GradCAM for.
            mean (float): Mean to normalize to image.
            std (float): Standard deviation to normalize the image.
            data_dir (str): Path of the directory to load the images from.
            df(pd.Dataframe): Dataframe with the image features.
            labels ([str]): All output labels for the model.
            selected_labels ([str]): All output labels we want to compute the GradCAM for.
            layer_name: Intermediate layer from the model we want to compute the GradCAM for.
        """
        
        og_img = original_img
        preprocessed_input = img_tensor
        predictions = model.predict(preprocessed_input)

        sorted_preds, sorted_labels = (list(reversed(t)) for t in zip(*sorted(zip(predictions[0], labels))))

        # Create a new figure with two subplots: one for the bar chart and one for the GradCAM
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Plotting the bar chart on the first subplot
        axs[1].barh(sorted_labels, sorted_preds)
        axs[1].set_title('Prediction Probabilities')
        axs[1].set_xlim([0, 1])

        # For the GradCAM on the second subplot
        i = 0  # 가장 높은 확률의 레이블 인덱스
        gradcam = grad_cam(model, preprocessed_input, i, layer_name)
        axs[0].imshow(og_img, cmap='bone')
        axs[0].imshow(gradcam, cmap='magma', alpha=min(0.5, sorted_preds[i]))
        axs[0].set_title(sorted_labels[i] + ": " + str(round(sorted_preds[i], 3)))
        axs[0].axis('off')

        # Display the figure in Streamlit
        st.pyplot(fig)



    BASE_PATH = "C:\\pjbm\\xray8\\chest_gc\\"
    SCRIPT_PATH = "C:\\pjbm\\xray8\\cchest_heatmap.py"

    @st.cache_data()
    def load_model_weights():
        # Create a temporary model to load the weights into
        temp_model = build_model()
        
        # Load the weights into the temporary model
        temp_model.load_weights(BASE_PATH + "nih_new\\pretrained_model.h5")
        
        # Return the weights from the temporary model
        return temp_model.get_weights()

    def load_weights_into_model(model):
        weights = load_model_weights()
        model.set_weights(weights)
        print("Loaded Weights")
        return model

    def build_model():
        labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

        train_df = pd.read_csv(BASE_PATH + "nih_new\\train-small.csv")

        class_pos = train_df.loc[:, labels].sum(axis=0)
        class_neg = len(train_df) - class_pos
        class_total = class_pos + class_neg

        pos_weights = class_pos / class_total
        neg_weights = class_neg / class_total
        print("Got loss weights")

        # create the base pre-trained model
        base_model = DenseNet121(weights=BASE_PATH + 'densenet.hdf5', include_top=False)
        print("Loaded DenseNet")

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # and a logistic layer
        predictions = Dense(len(labels), activation="sigmoid")(x)
        print("Added layers")

        model = Model(inputs=base_model.input, outputs=predictions)

        def get_weighted_loss(neg_weights, pos_weights, epsilon=1e-7):
            def weighted_loss(y_true, y_pred):
                # L(X, y) = −w * y log p(Y = 1|X) − w *  (1 − y) log p(Y = 0|X)
                # from https://arxiv.org/pdf/1711.05225.pdf
                loss = 0
                for i in range(len(neg_weights)):
                    loss -= (neg_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + 
                            pos_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
                
                loss = K.sum(loss)
                return loss
            return weighted_loss
        
        model.compile(optimizer='adam', loss=get_weighted_loss(neg_weights, pos_weights))
        print("Compiled Model")
        return model

    def load_weights_into_model(model):
        weights = load_model_weights()
        model.set_weights(weights)
        print("Loaded Weights")
        return model

    # Build and load weights into the model
    model_structure = build_model()
    model = load_weights_into_model(model_structure)


    # Set up the data necessary for processing
    IMAGE_DIR = BASE_PATH + 'nih_new\\images-small\\'
    df = pd.read_csv(BASE_PATH + "nih_new\\train-small.csv")

    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

    labels_to_show = labels

    mean, std = get_mean_std_per_batch(df)


    # Get the user-uploaded image and classify it
    if uploaded_file is not None:
        processed_img, processed_img_tensor = load_and_preprocess(uploaded_file, 320, 320, mean, std)
        compute_gradcam(model, processed_img, processed_img_tensor, mean, std, IMAGE_DIR, df, labels, labels_to_show)
    # else:
    #     processed_img, processed_img_tensor = load_and_preprocess(BASE_PATH + 'nih_new\\images-small\\00000284_005.png', 320, 320, mean, std)
    #     compute_gradcam(model, processed_img, processed_img_tensor, mean, std, IMAGE_DIR, df, labels, labels_to_show)
