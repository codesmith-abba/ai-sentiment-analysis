from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import joblib

categories = ['rec.sport.hockey', 'sci.med']

print('Downloading data...')
data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print("Finished getting data.")

texts = data.data # type: ignore
labels = data.target # type: ignore

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

print("Training model")

model = MultinomialNB()

model.fit(X_train, y_train)

print('Model trained')

joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model Saved")
