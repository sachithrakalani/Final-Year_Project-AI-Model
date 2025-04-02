import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

os.makedirs('models', exist_ok=True)\

data = pd.read_csv("data/deceptive-opinion.csv")

data['lable'] = data['deceptive'].map({'truthful': 0, 'deceptive': 1})

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['cleaned_review'] = data['text'].apply(clean_text)
