import streamlit as st
import numpy as np
import requests
import os
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
MAX_LENGTH = 150
EMBEDDING_DIM = 100
VOCAB_SIZE = 910915

# URLs for downloading the files
GLOVE_URL = 'https://drive.google.com/file/d/1Fq9zV2-o-Ej4bj3LPoCH9O8rDy5gyXoV/view?usp=sharing'
WEIGHTS_URL = 'https://drive.google.com/file/d/1oooHUpHgmSRZjY4Qt-sYNTN9QgUvmxOe/view?usp=sharing'
TOKENIZER_URL = 'https://drive.google.com/file/d/1f2CL3bxz1W8MVxF_zhlYqlAVrkod0KL7/view?usp=sharing'

# Function to download files from Google Drive
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Download the files if they don't exist locally
if not os.path.exists('glove.6B.100d.txt'):
    st.write("Downloading GloVe embeddings...")
    download_file(GLOVE_URL, 'glove.6B.100d.txt')

if not os.path.exists('model_weights.weights.h5'):
    st.write("Downloading model weights...")
    download_file(WEIGHTS_URL, 'model_weights.weights.h5')

if not os.path.exists('tokenizer.pickle'):
    st.write("Downloading tokenizer...")
    download_file(TOKENIZER_URL, 'tokenizer.pickle')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess text
def preprocess_text(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Function to load GloVe embeddings
def load_glove_embeddings(vocab_size, embedding_dim, tokenizer):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if word in tokenizer.word_index:
                idx = tokenizer.word_index[word]
                if idx < vocab_size:
                    embedding_matrix[idx] = coefs
    return embedding_matrix

# Load GloVe embeddings
embedding_matrix = load_glove_embeddings(VOCAB_SIZE, EMBEDDING_DIM, tokenizer)

# Define the model
input_layer = Input(shape=(MAX_LENGTH,))
embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False)(input_layer)
lstm_layer_1 = LSTM(64, return_sequences=True)(embedding_layer)
dropout_layer_1 = Dropout(0.5)(lstm_layer_1)
lstm_layer_2 = LSTM(32)(dropout_layer_1)
dense_layer = Dense(32, activation='relu')(lstm_layer_2)
dropout_layer_2 = Dropout(0.5)(dense_layer)
output_layer = Dense(1, activation='sigmoid')(dropout_layer_2)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load the model weights
model.load_weights('model_weights.weights.h5')

# Streamlit app
st.title("Sentiment Analysis")
st.write("Upload a review and get its sentiment prediction.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if user_input:
        preprocessed_input = preprocess_text(user_input, tokenizer, MAX_LENGTH)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to predict.")

st.write("Download the pre-trained model weights and GloVe embeddings from the links below if needed:")
st.write("[GloVe Embeddings](https://drive.google.com/file/d/1Fq9zV2-o-Ej4bj3LPoCH9O8rDy5gyXoV/view?usp=sharing)")
st.write("[Model Weights](https://drive.google.com/file/d/1oooHUpHgmSRZjY4Qt-sYNTN9QgUvmxOe/view?usp=sharing)")
st.write("[Tokenizer](https://drive.google.com/file/d/1f2CL3bxz1W8MVxF_zhlYqlAVrkod0KL7/view?usp=sharing)")
