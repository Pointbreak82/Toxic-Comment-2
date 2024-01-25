import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model using tf.saved_model.load
model_path = "toxicity_model"  
loaded_model = tf.saved_model.load(model_path)

st.title("Toxic Comment Classification")

# Create a text input for the user to input comments
user_input = st.text_area("Enter your comment:")

if st.button("Classify"):
    # Preprocess the user input using the TextVectorization layer
    user_input_sequence = loaded_model.signatures["serving_default"](
        input_1=tf.constant([user_input])
    )["text_vectorization"]

    # Pad the sequence to match the input size of the model
    user_input_padded = pad_sequences(user_input_sequence, maxlen=1800)

    # Make prediction
    predictions = loaded_model.signatures["serving_default"](
        input_1=user_input_padded
    )["dense_2"] > 0.5

    # Display the predictions
    st.subheader("Predictions:")
    for i, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        st.write(f"{col:<{20}}: {predictions[0][i]:<{5}}")
