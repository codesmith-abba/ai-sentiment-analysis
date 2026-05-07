import joblib

# Load model + vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

while True:
    text = input("Enter text (or type quit): ")

    if text.lower() == 'quit':
        break

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    if prediction[0] == 1:
        print("Positive")
    else:
        print("Negative")