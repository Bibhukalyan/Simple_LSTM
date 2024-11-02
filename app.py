import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

#load the model and tokenizer
model = tf.keras.models.load_model('hamlet_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    #tokenize the input text
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] ##ensure the sequence length matches the max sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #predict the next word
    predicted = model.predict(token_list, verbose=0)
    #get the index of the predicted word
    predicted_index = np.argmax(predicted, axis=1)
    #get the predicted word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None

#streamlit app
st.title('Shakespeare Text Generator')
input_text = st.text_input('Enter some text')
if st.button('Predict'):
    max_sequence_len = model.input_shape[1]+1
    predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Predicted word: {predicted_word}')
    
