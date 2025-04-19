import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

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

vectrorizer = TfidfVectorizer(max_features=5000)
X = vectrorizer.fit_transform(data['cleaned_review']).toarray()
y = data['lable']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

joblib.dump(model, 'models/fake_review_model.pkl')
joblib.dump(vectrorizer, 'models/tfidf_vectorizer.pkl')
print("Model and Vectorizer exported to 'models/' directory!")
