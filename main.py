import sys
from Plot_Classifier import plot_classification
from Text_Extractor import text_extractor 
from LSTM_seq2seq_world_level_inference import seq2seq_inference

#####################
# System's settings #
#####################

path_to_classifier_model = './models/Classification_model.h5'
path_to_encoder = './models/encoder_model.h5'
path_to_decoder = './models/decoder_model.h5'
path_to_images = "./shared_data/"
images_list = [path_to_images + "test_line_asc.png", path_to_images + "test2_line_asc.png"]

classification_results = plot_classification(images_list, path_to_classifier_model)

input_sequences = []

for res in classification_results:
    plot_type = res.split("_")[0]
    try: trend = res.split("_")[1]
    except: trend = ""
    temp_sentence = plot_type + " " + trend
    input_sequences.append(temp_sentence)

caption_results = seq2seq_inference(input_sequences, path_to_encoder, path_to_decoder)

print(caption_results)
