from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from train import vectrorizer

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

model = joblib.load('models/fake_review_model.pkl')
vectrorizer = joblib.load('models/tfidf_vectorizer.pkl')

# # Text preprocessing function
# def clean_text(text):
#     if not isinstance(text, str):  # Handle non-string inputs'
#         return ""
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\W', ' ', text)  # Remove special characters
#     text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
#     text = re.sub(r'\d+', '', text)  # Remove numbers
#     words = word_tokenize(text)  # Tokenize the text
#     stop_words = set(stopwords.words('english'))  # Load stopwords
#     words = [word for word in words if word not in stop_words]  # Remove stopwords
#     return ' '.join(words)

def clean_text(text):
    if not isinstance(text,str):
        return " "
    text = text.lower()
    text = re.sub(r'\W', ' ',text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
