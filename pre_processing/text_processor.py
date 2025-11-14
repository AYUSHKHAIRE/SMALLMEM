import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import emoji

class TextProcessor:
    def __init__(self):
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _clean_text(self, text):
        text = contractions.fix(text)
        text = emoji.demojize(text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\W+', ' ', text)
        return text.strip()

    def _tokenize_sentences(self, text):
        return word_tokenize(text)

    def _remove_stop_words(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def _apply_lemmatizer(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def process_text(self, text):
        clean_text = self._clean_text(text)
        tokens = self._tokenize_sentences(clean_text)
        filtered = self._remove_stop_words(tokens)
        lemmatized = self._apply_lemmatizer(filtered)
        return lemmatized
