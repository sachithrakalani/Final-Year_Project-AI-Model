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

def predict_review(review):
    cleaned_review = clean_text(review)
    vectrorized_review = vectrorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(vectrorized_review)
    return "Fake Review" if prediction[0] == 1 else "Real Review"

