# Product Review Helpfulness Prediction Using NLP

## 1. Project Overview

This project demonstrates a basic Natural Language Processing (NLP) and machine-learning workflow that classifies a product review as **likely helpful** or **likely unhelpful**.

The implementation uses:

- **TF-IDF** to convert review text into numerical features.
- **Logistic Regression** to classify the features.
- **Pandas** to load the CSV dataset.
- **scikit-learn** for data splitting, training, and evaluation.
- **Matplotlib** to visualize the confusion matrix.

## 2. Problem Statement

Online stores contain many customer reviews, but reviews vary in how much useful information they provide. A detailed review might explain product performance after regular use and identify a limitation, while a short review might simply say “Good product.”

This project treats the task as binary text classification:

- `1` — likely helpful
- `0` — likely unhelpful

The model takes review text as input and predicts one of these labels. In this demonstration, the labels are assigned during synthetic data generation. Specific, descriptive reviews tend to be labelled likely helpful, while vague or short opinions tend to be labelled likely unhelpful. This is a simplified teaching rule, not a universal definition of helpfulness.

## 3. Objectives

1. Create a small dataset of sample product reviews.
2. Convert review text into numeric features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Evaluate the model on a held-out test set.
5. Report accuracy, precision, recall, F1-score, and a confusion matrix.
6. Demonstrate prediction on a new example review.
7. Discuss the limitations of synthetic data and avoid unsupported real-world claims.

## 4. Project Files

| File                            | Purpose                                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------- |
| `review_helpfulness.py`         | Main program: loads data, trains the model, evaluates it, and predicts a sample review. |
| `synthetic_product_reviews.csv` | Generated dataset containing review text and binary labels.                             |
| `results.txt`                   | Saved metrics and confusion-matrix values from the experiment.                          |
| `confusion_matrix.png`          | Visualization of actual versus predicted classes.                                       |

Keep `review_helpfulness.py` and `synthetic_product_reviews.csv` in the same folder. The script loads the CSV from its own directory.

## 5. Dataset Description

The CSV file contains two columns:

| Column    | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| `review`  | Text of a generated product review.                                |
| `helpful` | Target label: `1` for likely helpful and `0` for likely unhelpful. |

The supplied dataset contains **101 unique review examples**:

- 60 labelled likely helpful (`1`)
- 41 labelled likely unhelpful (`0`)

Examples mention everyday products such as headphones, phone chargers, water bottles, laptop stands, desk lamps, keyboards, backpacks, and speakers.

## 6. Machine-Learning Method

### 6.1 Train/test split

The dataset is divided into:

- **75 training examples** (approximately 75%)
- **26 test examples** (approximately 25%)

The split uses `random_state=42` for reproducibility and stratification to preserve approximately similar class proportions. The test set is not used to fit the model.

### 6.2 TF-IDF

TF-IDF means **Term Frequency–Inverse Document Frequency**. It converts text into numerical features. Term frequency reflects how often a word appears in a review, while inverse document frequency reduces the influence of words that appear in many reviews. This can give more weight to distinctive terms.

The implementation lowercases text and removes common English stop words. TF-IDF does not understand language like a human; it represents statistical patterns in the text.

### 6.3 Logistic Regression

Logistic Regression is a supervised classification algorithm. It learns associations between the TF-IDF features and the labels supplied during training. It is fast, straightforward, and suitable as a baseline for a small text-classification task.

### 6.4 Evaluation metrics

- **Accuracy:** Fraction of test reviews classified correctly.
- **Precision:** Among reviews predicted as a class, the fraction that actually belongs to that class.
- **Recall:** Among reviews that actually belong to a class, the fraction correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts correct predictions and errors for each actual/predicted class combination.

Accuracy alone is not enough to describe performance, so the class-specific metrics and confusion matrix should also be reviewed.

## 7. Requirements

Install Python 3.9 or later. The project requires:

- `pandas`
- `scikit-learn`
- `matplotlib`

No GPU or internet connection is needed after the project files have been downloaded.

## 8. Installation and Execution

### Step 1: Extract the project

Extract the ZIP file or place the project files in one folder. At minimum, ensure that these two files are together:

```text
review_helpfulness.py
synthetic_product_reviews.csv
```

### Step 2: Open a terminal

Open a terminal or command prompt in the project folder. In VS Code, choose **Terminal → New Terminal**.

### Step 3: Install dependencies

```bash
python -m pip install pandas scikit-learn matplotlib
```

If your system uses `python3`, run:

```bash
python3 -m pip install pandas scikit-learn matplotlib
```

### Step 4: Run the program

```bash
python review_helpfulness.py
```

Or:

```bash
python3 review_helpfulness.py
```

### Step 5: Inspect the output

The program prints the dataset size, training/test sizes, accuracy, classification report, and a prediction for an illustrative review. It also saves a confusion-matrix image in the `results` folder. A plot window may appear depending on your environment.

## 9. Results from the Supplied Experiment

Using the supplied dataset and fixed split, the recorded experiment produced:

| Metric                       | Result |
| ---------------------------- | -----: |
| Total examples               |    101 |
| Training examples            |     75 |
| Test examples                |     26 |
| Accuracy                     | 92.31% |
| Likely unhelpful — precision |   1.00 |
| Likely unhelpful — recall    |   0.82 |
| Likely unhelpful — F1-score  |   0.90 |
| Likely helpful — precision   |   0.88 |
| Likely helpful — recall      |   1.00 |
| Likely helpful — F1-score    |   0.94 |

The confusion matrix uses rows for actual classes `[unhelpful, helpful]` and columns for predicted classes `[unhelpful, helpful]`:

```text
[[ 9,  2],
 [ 0, 15]]
```

Interpretation:

- 9 unhelpful-labelled reviews were correctly predicted as unhelpful.
- 2 unhelpful-labelled reviews were predicted as helpful.
- 15 helpful-labelled reviews were correctly predicted as helpful.
- 0 helpful-labelled reviews were predicted as unhelpful.

**Interpretation warning:** These metrics are from a small synthetic test set. Since the generated text and labels follow related rules, the score may be optimistic. It must not be described as 92.31% accuracy on real product reviews.

## 10. Example Prediction

The program sends an illustrative review through the fitted model:

> “After three weeks, the headphones have clear sound, but the cable feels short.”

The model prints either `LIKELY HELPFUL` or `LIKELY UNHELPFUL`. This example demonstrates how to predict a new review. It is not part of the held-out test set and must not be included in the evaluation metrics.

## 11. Conclusion

This project demonstrates a complete, beginner-friendly NLP workflow. A synthetic review dataset is loaded, text is converted into TF-IDF features, Logistic Regression is trained, and predictions are evaluated using standard classification metrics.

The supplied experiment achieved **92.31% accuracy on 26 synthetic test examples**. This demonstrates how the pipeline behaves on the generated data, but does not prove that the model can predict real-world review helpfulness with the same accuracy.

The project is intended to be easy to run, explain, and reproduce for **23CSE401 – Fundamentals of AI**. Its main learning outcomes are understanding text vectorization, supervised classification, train/test evaluation, and the need to interpret metrics in the context of the data.
