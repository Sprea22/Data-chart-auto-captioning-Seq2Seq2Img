# import the necessary packages
import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def model_predict(input_image, model):
    labels = ['bar_plot_', 'bar_plot_asc', 'bar_plot_desc', 
        'line chart_no trend', 'line chart_ascendant', 'line chart_descendant', 
        'scatter_plot_', 'scatter_plot_asc', 'scatter_plot_desc']
        
    # Loading the image
    img = image.load_img(input_image, target_size=(192, 192))
    # Needed preproccesing on the input image
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data/255.0
    
    # Prediction image 
    image_prediction = model.predict(img_data)
    
    # Prediction result
    image_prediction_result = np.argmax(image_prediction[0])
    result = labels[image_prediction_result]     

    return result

#############
# DASHBOARD #
#############

def plot_classification(images_list, path_to_classifier_model):

    # Loading the classification model
    model_path = path_to_classifier_model
    model = load_model(model_path, compile=False)
    print('Model loaded.')
    model._make_predict_function()         
    print('Model loaded. Start serving...')

    predictions_list = []

    for input_image in images_list:

        # Classifying the input image
        prediction = model_predict(input_image, model)
        predictions_list.append(prediction)

    return predictions_list
    