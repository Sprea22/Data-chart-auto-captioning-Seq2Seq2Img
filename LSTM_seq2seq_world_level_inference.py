
import re
import pickle 
import string
import numpy as np
import pandas as pd
from string import digits
import matplotlib.pyplot as plt
from keras.models import load_model

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of linpth 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max linpth
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 52):
            stop_condition = True

        # Update the target sequence (of linpth 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence
    
def seq2seq_inference(input_sentences, path_to_encoder, path_to_decoder):

    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    # LOADING THE SAVED FILES DURING THE MODEL TRAINING PHASE #
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    global reverse_input_char_index, reverse_target_char_index
    global encoder_model, decoder_model
    global encoder_input_data, input_token_index, target_token_index

    pkl_files_dir = 'cached_files/'

    pkl_file = open(pkl_files_dir + 'reverse_input_char_index.pkl', 'rb')
    reverse_input_char_index= pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(pkl_files_dir + 'reverse_target_char_index.pkl', 'rb')
    reverse_target_char_index= pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(pkl_files_dir + 'input_token_index.pkl', 'rb')
    input_token_index= pickle.load(pkl_file)
    pkl_file.close()

    pkl_file = open(pkl_files_dir + 'target_token_index.pkl', 'rb')
    target_token_index= pickle.load(pkl_file)
    pkl_file.close()

    ### ### ### ### ### ### ### ### ##
    # LOADING THE PRE TRAINED MODELS #
    ### ### ### ### ### ### ### ### ##
    encoder_model = load_model(path_to_encoder)
    decoder_model = load_model(path_to_decoder)

    ### ### ### ### ####
    # MODEL INFERENCES #
    ### ### ### ### ####
    # Encoding the input sentences
    encoder_input_data = np.zeros((len(input_sentences), 7), dtype='float32')
    for i, input_text in enumerate(input_sentences):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = input_token_index[word]
            
    decoded_sentences = []
    for seq_index in range(0,len(input_sentences)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        # decode_sequence is the inference function of the model
        decoded_sentence = decode_sequence(input_seq)
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences