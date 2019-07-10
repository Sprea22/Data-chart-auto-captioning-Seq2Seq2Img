
import re
import string
import numpy as np
import pandas as pd
from string import digits
import matplotlib.pyplot as plt
from keras.models import load_model

def sentences_pre_processing(lines):
    # Pre processing on the input sentences
    lines.inp=lines.inp.apply(lambda x: x.lower())
    lines.out=lines.out.apply(lambda x: x.lower())

    # Replying some special chars from the input sentences
    lines.inp=lines.inp.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    lines.out=lines.out.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))

    # Removing punctuations from the input senteces
    exclude = set(string.punctuation)
    lines.inp=lines.inp.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.out=lines.out.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # Removing digits from the input senteces
    remove_digits = str.maketrans('', '', digits)
    lines.inp=lines.inp.apply(lambda x: x.translate(remove_digits))
    lines.out=lines.out.apply(lambda x: x.translate(remove_digits))

    # Adding 'START_ ' token at the beginning of each output sentence
    # Adding ' _END' token at the end of each output sentence
    lines.out = lines.out.apply(lambda x : 'START_ '+ x + ' _END')

    return lines

def data_encoding(lines):
    # Creating the input sentences dictionary
    all_inp_words=set()
    for inp in lines.inp:
        for word in inp.split():
            if word not in all_inp_words:
                all_inp_words.add(word)
    input_words = sorted(list(all_inp_words))
    num_encoder_tokens = len(all_inp_words) + 1
    input_token_index = dict([(word, i) for i, word in enumerate(input_words, 1)])

    # Creating the output sentences dictionary
    all_out_words=set()
    for out in lines.out:
        for word in out.split():
            if word not in all_out_words:
                all_out_words.add(word)
    target_words = sorted(list(all_out_words))
    num_decoder_tokens = len(all_out_words) + 1
    target_token_index = dict([(word, i) for i, word in enumerate(target_words, 1)])

    # Intiailizing the encoder/decoder arrays
    encoder_input_data = np.zeros((len(lines.inp), 7), dtype='float32')

    # Representing each input/output/target sentence using an array.
    # !!! Input length right now is set to 7 and output to 16. 
    # !!! In order to change it just modify the numbers few rows above
    # EX ---> [5, 7, 3, 1, 0, 0, 0]
    for i, (input_text, target_text) in enumerate(zip(lines.inp, lines.out)):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[i, t] = input_token_index[word]


    # Loading the pre trained models of the encoder and decoder
    return encoder_input_data, input_token_index, target_token_index

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
    # Reading the input - output sentences
    lines= pd.read_table('test.txt', sep="___", names=['inp', 'out'])

    ### ### ### ### ### ####
    # INPUT PRE PROCESSING #
    ### ### ### ### ### ####
    lines = sentences_pre_processing(lines)

    ### ### ### ### ### ### #####
    # ANALYZING INPUT SENTENCES #
    ### ### ### ### ### ### #####
    global reverse_input_char_index, reverse_target_char_index, encoder_model, decoder_model, encoder_input_data, input_token_index, target_token_index

    encoder_input_data, input_token_index, target_token_index = data_encoding(lines)

    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    ### ### ### ### ### ### ### ### ##
    # LOADING THE PRE TRAINED MODELS #
    ### ### ### ### ### ### ### ### ##
    encoder_model = load_model(path_to_encoder)
    decoder_model = load_model(path_to_decoder)

    ### ### ### ### ####
    # MODEL INFERENCES #
    ### ### ### ### ####
    decoded_sentences = []
    for input_seq in input_sentences:
        encoded_input_sentence = np.zeros(7, dtype='float32')
        for t, word in enumerate(input_seq.split()):
            encoded_input_sentence[t] = input_token_index[word]

        # decode_sequence is the inference function of the model
        decoded_sentence = decode_sequence(encoded_input_sentence)
        decoded_sentences.append(decoded_sentence)
        #print('-')
        #print('Input sentence:', input_seq)
        #print('Encoded Input sentence:', encoded_input_sentence)
        #print('Decoded Output sentence:', decoded_sentence)
    
    return decoded_sentences