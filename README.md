# AI Sentiment Analysis

A simple NLP-based sentiment analysis project built with Python and machine learning.

This project classifies text as positive or negative using TF-IDF vectorization and the Naive Bayes algorithm.

---

## Features

- Text preprocessing
- TF-IDF vectorization
- Sentiment classification
- Model training and evaluation
- Interactive prediction system
- Model saving/loading

---

## Tech Stack

- Python
- scikit-learn
- NLTK
- Joblib

---

## Project Structure

```text
ai-sentiment-analysis/
│
├── model/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses a real-world text dataset from scikit-learn:

- rec.sport.hockey
- sci.med

The dataset is downloaded automatically during training.

---

## Model Pipeline

1. Load text dataset
2. Convert text into numerical vectors using TF-IDF
3. Split dataset into training and testing sets
4. Train model using Multinomial Naive Bayes
5. Evaluate model accuracy
6. Save trained model
7. Predict sentiment from user input

---

## Installation

Clone the repository:

```bash
git clone https://github.com/codesmith-abba/ai-sentiment-analysis.git
cd ai-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python train.py
```

---

## Run Prediction App

```bash
python predict.py
```

Example:

```text
Enter text: I love this app
Positive

Enter text: This is terrible
Negative
```

---

## Accuracy

Current model accuracy:

```text
97%
```

---

## Future Improvements

- Use larger sentiment datasets
- Add deep learning models
- Build Flask/FastAPI API
- Deploy as web app
- Add multi-class sentiment detection
- Integrate transformer models

---

## Author

Built by CodeSmith