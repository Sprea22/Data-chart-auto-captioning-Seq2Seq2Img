import sys
import time
from Plot_Classifier import plot_classification
from Text_Extractor import text_extractor 
from LSTM_seq2seq_world_level_inference import seq2seq_inference
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
 
#####################
# SYSTEM'S SETTINGS #
#####################
path_to_classifier_model = './models/classification_model.h5'
path_to_encoder = './models/encoder_model.h5'
path_to_decoder = './models/decoder_model.h5'
path_to_images = "./shared_data/"

##########################
# DATA CHART IMAGES LIST #
##########################
images_list = []
for img in os.listdir(path_to_images):
    images_list.append(path_to_images + img)

################################
# CLASSIFYING THE INPUT IMAGES #
################################
print("------- Classifying the input images..\n")
classification_results = plot_classification(images_list, path_to_classifier_model)

# Formatting the classification results in order to feed the LSTM seq2seq model
print("------- Formatting the classification results..\n")
input_sequences = []
for res in classification_results:
    plot_type = res.split("_")[0]
    try: trend = res.split("_")[1]
    except: trend = ""
    temp_sentence = plot_type + " " + trend
    input_sequences.append(temp_sentence)

###############################################
# EXTRACTING THE TEXTUAL INFO FROM THE IMAGES #
###############################################
print("------- Extracting the textual information from the images..\n")
text_results = text_extractor(images_list)
print(text_results)

##################################################
# CAPTION GENERATION THROUGH LSTM2 seq2seq MODEL #
##################################################
print("------- Generating the caption..\n")
caption_results = seq2seq_inference(input_sequences, path_to_encoder, path_to_decoder)

##########################
# DISPLAYING THE RESULTS #
##########################
print("------- SUCCESS! \n")
print(caption_results)

for i in range(0,3):
    print("\n#### About the image number " + str(i) + " ####")
    print("TITLE: ", text_results[i])
    print("CAPTION: ", caption_results[i])