import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
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

model = load_model('static/IMDB_sentiment_analyzer__model (1).h5')

with open('static/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Streamlit app
st.title("Sentiment Analysis Web App")

st.write("Enter a movie review below and get the sentiment prediction:")

# Input text
user_input = st.text_area("Review", "")

if st.button("Predict Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"The sentiment of the review is: {sentiment}")
    else:
        st.write("Please enter a review.")
