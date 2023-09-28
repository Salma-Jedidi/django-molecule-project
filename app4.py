# import core libraries 
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import random as rn
import matplotlib.pyplot as plt
import tensorflow as tf

# load Keras libraries
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# set random seed
seed = 777
np.random.seed(seed)
rn.seed(seed)
np.seterr(divide='ignore')

st.title('Molecule Generator')
d1 = st.file_uploader("Upload char_to_int File", type=["json"])
d2 = st.file_uploader("Upload int_to_char File", type=["json"])
latent_data = st.file_uploader('Upload SMILES file')

#load model
model = tf.keras.models.load_model('LSTM_model.hdf5')

def start_process():
    # create the encoder model from the previously trained model
    encoder_model = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)

    # create a model for mapping from the latent space to the input states of the decoder LSTM model
    latent_input = Input(shape=(128, ))
    state_h = model.layers[5](latent_input)
    state_c = model.layers[6](latent_input)
    latent_to_states_model = Model(latent_input, [state_h, state_c])

    # define the stateful decoder model
    decoder_inputs = Input(batch_shape=(1, 1, 45))
    decoder_lstm = LSTM(256, return_sequences=True, stateful=True)(decoder_inputs)
    decoder_outputs = Dense(45, activation='softmax')(decoder_lstm)
    gen_model = Model(decoder_inputs, decoder_outputs)

    for i in range(1,3):
        gen_model.layers[i].set_weights(model.layers[i+6].get_weights())

    def load_data(data):
        if data is not None:
            content = data.getvalue().decode("utf-8")
            smiles = [r.rstrip() for r in content.split('\n')]
            return np.array(smiles)
        return None


    def load_dictionaries(input_dict):
        if input_dict is not None:
            content = input_dict.getvalue().decode("utf-8")
            new_dict = json.loads(content)
            return new_dict
        return None

    latent = load_data(latent_data) 

    char_to_int = load_dictionaries(d1)
    int_to_char = load_dictionaries(d2)
    n_vocab = len(char_to_int)

    # create our Softmax sampling function
    def sample_with_temp(preds, sampling_temp):
        streched = np.log(preds) / sampling_temp
        streched_probs = np.exp(streched) / np.sum(np.exp(streched))
        return np.random.choice(range(len(streched)), p=streched_probs)

    #create a function to generate new smiles from the latent space
    def sample_smiles(latent, n_vocab, sampling_temp):
        #decode the latent states and set the initial state of the LSTM cells
        states = latent_to_states_model.predict(latent)
        gen_model.layers[1].reset_states(states=[states[0], states[1]])
        # define the input character
        startidx = char_to_int["!"]
        samplevec = np.zeros((1,1,n_vocab))
        samplevec[0,0,startidx] = 1
        sequence = ""
        # loop to predict the next smiles character
        for i in range(101):
            preds = gen_model.predict(samplevec)[0][-1]
            if sampling_temp == 1.0:
              sampleidx = np.argmax(preds)
            else:
              sampleidx = sample_with_temp(preds, sampling_temp)
            samplechar = int_to_char[str(sampleidx)]
            if samplechar != "E":
                sequence += samplechar
                samplevec = np.zeros((1,1,n_vocab))
                samplevec[0,0,sampleidx] = 1
            else:
                break
        return sequence



    # function to transform our smiles data into a supervised learning dataset
    def vectorize(smiles, embed, n_vocab):
        one_hot = np.zeros((smiles.shape[0], embed, n_vocab), dtype=np.int8)
        for i, smile in enumerate(smiles):
            # encode the start
            one_hot[i,0,char_to_int["!"]] = 1
            #encode the smiles characters
            for j, c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            # encode the end of the smiles string
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        # return two items, one for input and one for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]

    # create our latent space input tensor
    X, _ = vectorize(latent, 101, n_vocab)

    # create our latent space for generating molecules using our encoder model
    latent_space = encoder_model.predict(X)

    # function to generate smiles around a latent vector
    def generate(latent_seed, sampling_temp, scale, quant):
      samples, mols = [], []
      for i in range(quant):
        latent_vec = latent_seed + scale*(np.random.randn(latent_seed.shape[1]))
        out = sample_smiles(latent_vec, n_vocab, sampling_temp)
        mol = Chem.MolFromSmiles(out)
        if mol:
          mols.append(mol)
          samples.append(out)
      return mols, samples

    # generate molecules based off our sampling dataset's latent space
    gen_mols, gen_smiles = [], []
    for i in range(latent.shape[0] - 1):
      latent_seed = latent_space[i:i+1]
      sampling_temp = rn.uniform(0.5, 1.2)
      scale = 0.5
      quantity = 25
      mols, smiles = generate(latent_seed, sampling_temp, scale, quantity)
      gen_mols.extend(mols)
      gen_smiles.extend(smiles)
      moles, smiles = [], []

    df = pd.DataFrame({'SMILES': gen_smiles})
    st.dataframe(df)

    mols = [Chem.MolFromSmiles(smile) for smile in gen_smiles]
    st.image(Draw.MolsToGridImage(mols, molsPerRow=10))
if st.button('Start'):
    start_process()