import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import joblib

categories = ['rec.sport.hockey', 'sci.med']

print('Downloading data...')
data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

print("Finished Getting data")

texts = data.data # type: ignore
labels = data.target # type: ignore

# df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)
y = labels

# print(encoder.classes_)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

print("Training Model")

model = MultinomialNB()

model.fit(X_train, y_train)

print('model trained')

joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model Saved")

predictions = model.predict(X_test)

# print(predictions)
# print(y_test)

# accuracy = accuracy_score(y_test, predictions)

# print(f"Accuracy: {accuracy * 100:.2f}%")

# sample_text = ["I really love this product"]

# sample_vector = vectorizer.transform(sample_text)

# prediction = model.predict(sample_vector)

# print("Prediction Result:", prediction)

# sentiment = encoder.inverse_transform(prediction)

# print("Sentiment:", sentiment[0])

# sample_text2 = ["Worst product ever"]
# sample_vector2 = vectorizer.transform(sample_text2)
# prediction2 = model.predict(sample_vector2)
# sentiment2 = encoder.inverse_transform(prediction2)
# print("Sentiment 2:", sentiment2[0])


# print(X.toarray()) # type: ignore
# print(vectorizer.get_feature_names_out())
# print(y)

