# SMS Spam Detection System

A Machine Learning-based web application that classifies SMS messages as **Spam** or **Normal** using Natural Language Processing (NLP) and Python.

## Project Overview

The SMS Spam Detection System helps identify unwanted and potentially harmful SMS messages using machine learning algorithms. Users can enter a message and instantly receive a prediction with a confidence score.

The application also provides message history, model evaluation metrics, and an interactive dashboard.

## Features

- Real-time SMS spam detection
- Text preprocessing and TF-IDF vectorization
- Machine Learning model comparison
- Logistic Regression and Multinomial Naive Bayes
- Prediction confidence score
- Interactive Streamlit dashboard
- Message history stored using SQLite
- CSV export of detection history
- Model evaluation and confusion matrix
- Classification report

## Machine Learning Models

Two machine learning algorithms were evaluated:

1. Multinomial Naive Bayes
2. Logistic Regression

Logistic Regression was selected based on its F1-score.

## Model Performance

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 97.31% |
| Precision | 88.39% |
| Recall | 91.95% |
| F1 Score | 90.13% |

Dataset size: 5,572 SMS messages.

Training/testing split: 80% / 20%.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK / NLP concepts
- TF-IDF
- SQLite
- Matplotlib / Seaborn

## Project Structure

```text
spam-detector/
│
├── app.py
├── train.py
├── model_metrics.json
├── README.md
├── .gitignore
│
├── model.pkl          # Generated locally
├── vectorizer.pkl     # Generated locally
└── spam_history.db    # Generated locally
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ruchitbaghlahub/sms-spam-detector.git
cd sms-spam-detector
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

### 3. Train the model

```bash
python train.py
```

This generates the trained model and TF-IDF vectorizer files.

### 4. Run the application

```bash
python -m streamlit run app.py
```

## Application Pages

- Dashboard
- Detect Message
- History
- Model Evaluation
- About Project

## Future Improvements

- Deploy the application online
- Add more machine learning algorithms
- Improve text preprocessing
- Add model explainability
- Build a REST API for predictions

## Author

**Ruchit Baghla**

GitHub: [@ruchitbaghlahub](https://github.com/ruchitbaghlahub)

---

If you find this project useful, consider giving it a ⭐ on GitHub.
