import nltk
import string
import pickle
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('tfidf.pickle', 'rb'))
model = pickle.load(open('model.pickle', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def vectorize_word(text):
    return tfidf.transform([text])


def predict(vectorized_sentence):
    return model.predict(vectorized_sentence)[0]
