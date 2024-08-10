import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import requests
import contractions
import pickle

def clean_text(review):
    review = contractions.fix(review)
    review = review.lower()
    review = re.sub(r'<.*?>', ' ', review)
    review = re.sub(r'[^a-zA-Z\s]', ' ', review)
    review = re.sub(r'\s+', ' ', review).strip()
    return review


def predict_sentiment(review):
  # clean, tokenize and pad the review
  clean_review = clean_text(review)
  sequence = tokenizer.texts_to_sequences([clean_review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment


url = 'https://drive.google.com/uc?export=download&id=1T2_OP1lH_uXZ4lfxZ458GTsTWXbGxm7B'
response = requests.get(url)
with open("model.h5", "wb") as f:
    f.write(response.content)

model = load_model("model.h5")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Streamlit app
st.title("Welcome to SentiMeter!")

st.write("Enter a movie review below and get the sentiment prediction:")

# Input text
user_input = st.text_area("Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"The sentiment of the review is: {sentiment}")
    else:
        st.write("Please enter a review.")
