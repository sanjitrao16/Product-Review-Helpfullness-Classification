"""
Review Helpfulness Prediction using TF-IDF and Logistic Regression.
This project uses a small SYNTHETIC educational dataset created for demonstration.
It does not prove real-world helpfulness prediction performance.
Requirements: pandas, scikit-learn, matplotlib
Run: python review_helpfulness.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_FILE = Path(__file__).with_name("synthetic_product_reviews.csv")
OUT_DIR = Path(__file__).with_name("results")
OUT_DIR.mkdir(exist_ok=True)

# Load the supplied synthetic dataset.
df = pd.read_csv(DATA_FILE)
X = df["review"].astype(str)
y = df["helpful"].astype(int)  # 1 = likely helpful, 0 = likely unhelpful

# Keep a separate test set that is not used for model fitting.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# TF-IDF turns words into numeric features; Logistic Regression classifies them.
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Total reviews: {len(df)}")
print(f"Training reviews: {len(X_train)}")
print(f"Test reviews: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["Likely unhelpful", "Likely helpful"],
    zero_division=0
))

cm = confusion_matrix(y_test, predictions, labels=[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Unhelpful", "Helpful"]
).plot(values_format="d")
plt.title("Review Helpfulness Prediction — Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
plt.show()

# Try an example review. This prediction is a demo, not an evaluation result.
example = "After three weeks, the headphones have clear sound, but the cable feels short."
label = model.predict([example])[0]
print("\nExample review:", example)
print("Prediction:", "LIKELY HELPFUL" if label == 1 else "LIKELY UNHELPFUL")
