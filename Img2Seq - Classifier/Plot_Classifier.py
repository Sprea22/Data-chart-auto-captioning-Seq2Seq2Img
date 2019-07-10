# import the necessary packages
from PIL import Image
import argparse
import cv2
import os
import numpy as np
import json
from keras.preprocessing import image
from keras.models import load_model

def formal_description(input_image, prediction):
    # Generating the formal description through the JSON object
    plot_type = prediction.split("_")[0]
    try: trend = prediction.split("_")[2]
    except: trend = ""

    formal_description = {
        "filename" : input_image,
        "plot_type": plot_type,
        "trend": trend,
    }

    # Saving the JSON Object in the right path
    formal_description_name = input_image.split(".")[0] + ".json"
    if(not(os.path.isfile(formal_description_name))):
        file = open(formal_description_name, 'w+')
    with open(formal_description_name, 'w') as outfile:
        json.dump(formal_description, outfile)

def model_predict(input_image, model):
    labels = ['bar_plot_', 'bar_plot_asc', 'bar_plot_desc', 
        'line_plot_', 'line_plot_asc', 'line_plot_desc', 
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

path_to_images = "./../shared_data/"
images_list = [path_to_images + "test_line_asc.png", path_to_images + "test_weather.jpg"]

# Loading the classification model
model_path = "Classification_model.h5"
model = load_model(model_path)
print('Model loaded.')
model._make_predict_function()         
print('Model loaded. Start serving...')

for input_image in images_list:

    # Extracting all the text from an input image
    full_text = text_extractor("full_text", input_image)

    # Classifying the input image
    prediction = model_predict(input_image, model)

    # Saving the generated formal description in a JSON file
    formal_description(input_image, prediction)