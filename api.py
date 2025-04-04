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

