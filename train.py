import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

os.makedirs('models', exist_ok=True)