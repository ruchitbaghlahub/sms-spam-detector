
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).parent

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading SMS Spam Dataset...")

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "message"]
)

df = df.dropna()

df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label_num"])

X = df["message"].astype(str)
y = df["label_num"].astype(int)

print("Total Messages:", len(df))
print("Spam Messages:", int(y.sum()))
print("Normal Messages:", int((y == 0).sum()))

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# 5. DEFINE ML MODELS
# ==========================================

models = {
    "Multinomial Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = {}
trained_models = {}

best_model = None
best_model_name = None
best_f1 = -1

print("\nStarting Model Training...\n")

for name, classifier in models.items():

    print("Training:", name)

    classifier.fit(X_train_vec, y_train)

    predictions = classifier.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    results[name] = {
        "accuracy": round(float(accuracy * 100), 2),
        "precision": round(float(precision * 100), 2),
        "recall": round(float(recall * 100), 2),
        "f1_score": round(float(f1 * 100), 2),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Spam"],
            output_dict=True,
            zero_division=0
        )
    }

    trained_models[name] = classifier

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")
    print("-" * 40)

    # Select best model using F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = classifier
        best_model_name = name

# ==========================================
# 7. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    BASE_DIR / "model.pkl"
)

joblib.dump(
    vectorizer,
    BASE_DIR / "vectorizer.pkl"
)

# ==========================================
# 8. SAVE EVALUATION METRICS
# ==========================================

evaluation_data = {
    "best_model": best_model_name,
    "dataset_size": int(len(df)),
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "models": results
}

with open(
    BASE_DIR / "model_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        evaluation_data,
        file,
        indent=4
    )

# ==========================================
# 9. FINAL RESULTS
# ==========================================

print("\n" + "=" * 45)
print("MODEL TRAINING COMPLETED")
print("=" * 45)

print("Best Model:", best_model_name)
print("Best F1 Score:", round(best_f1 * 100, 2), "%")

print("\nSaved Files:")
print("1. model.pkl")
print("2. vectorizer.pkl")
print("3. model_metrics.json")

print("\nAll models compared successfully!")
