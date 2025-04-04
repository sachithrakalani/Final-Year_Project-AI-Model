from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

model = joblib.load('models/fake_review_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def clean_text(text):
    if not isinstance(text, str):  # Handle non-string inputs
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Load stopwords
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

def predict_review(review):
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(vectorized_review)
    return "Fake Review" if prediction[0] == 1 else "Real Review"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'No review provided'}), 400

    review = data['review']
    prediction = predict_review(review)
    return jsonify({'review': review, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)