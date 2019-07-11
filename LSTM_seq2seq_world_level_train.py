
import re
import string
import keras
import numpy as np
import pandas as pd
from string import digits
import matplotlib.pyplot as plt

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

# Reading the input - output sentences
lines= pd.read_table('cached_files/test.txt', sep="___", names=['inp', 'out'])

### ### ### ### ### ####
# INPUT PRE PROCESSING #
### ### ### ### ### ####

lines = sentences_pre_processing(lines)

### ### ### ### ### ### #####
# ANALYZING INPUT SENTENCES #
### ### ### ### ### ### #####

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

# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

# Intiailizing the encoder/decoder arrays
encoder_input_data = np.zeros((len(lines.inp), 7), dtype='float32')
decoder_input_data = np.zeros((len(lines.out), 16), dtype='float32')
decoder_target_data = np.zeros((len(lines.out), 16, num_decoder_tokens), dtype='float32')

# Representing each input/output/target sentence using an array.
# !!! Input length right now is set to 7 and output to 16. 
# !!! In order to change it just modify the numbers few rows above
# EX ---> [5, 7, 3, 1, 0, 0, 0]
for i, (input_text, target_text) in enumerate(zip(lines.inp, lines.out)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

import pickle 

# write the dicts to files
dictionary_list = {
                    "reverse_input_char_index"      : reverse_input_char_index, 
                    "reverse_target_char_index"     : reverse_target_char_index, 
                    "input_token_index"             : input_token_index, 
                    "target_token_index"            : target_token_index
                    }

for to_save_dict in dictionary_list:
    output_dict = open("cached_files/" + to_save_dict + '.pkl', 'wb')
    pickle.dump(dictionary_list[to_save_dict], output_dict)
    output_dict.close()
    

### ### ### ### ### ### ### ### #### 
# ENCODER-DECODER MODEL DEFINITION #
### ### ### ### ### ### ### ### #### 

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model

embedding_size = 50

# Encoder model structure
encoder_inputs = Input(shape=(None,))
en_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# Discard the `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Decoder model structure
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dex = Embedding(num_decoder_tokens, embedding_size)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Encoder-Decoder model structure

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

'''
earlystop = keras.callbacks.EarlyStopping(monitor = 'val_acc',
                                            min_delta = 0.0001, 
                                            patience = 100,
                                            verbose = 1,
                                            mode='auto',
                                            restore_best_weights=True)
callbacks_list = [earlystop]
'''

# Fit the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=5,
          epochs=500,
          validation_split=0.20)

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

# Create sampling model
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2= dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

### ### ### ### ### ### ### #
# SAVING THE TRAINED MODELS #
### ### ### ### ### ### ### #

encoder_model.save('models/encoder_model.h5')  
decoder_model.save('models/decoder_model.h5')  

