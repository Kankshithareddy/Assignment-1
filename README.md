# Assignment-1
# Project: FIFA Player Classification using SVM
📌 Objective

This project applies a Support Vector Machine (SVM) classifier to predict whether a FIFA player is above average or not.

A player is considered above average if their Overall rating ≥ 80.

📂 Dataset

File: fifa_data.csv

Contains detailed information about FIFA players.

Only numeric features were used for training.

Target column created:

Above_Avg = 1 → Overall ≥ 80

Above_Avg = 0 → Overall < 80

⚙️ Methodology

Data Preprocessing

Loaded CSV data into pandas DataFrame.

Dropped unnecessary columns (Overall, ID, Unnamed: 0).

Handled missing values by replacing them with 0.

Train-Test Split

Training: 80%

Testing: 20%

Model

Algorithm: Support Vector Classifier (SVC)

Kernel: RBF (Radial Basis Function)

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score
💻 Code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
file_path = "/content/fifa_data.csv"
df = pd.read_csv(file_path, encoding="latin1")

# Target: Is player above average (Overall >= 80)?
df["Above_Avg"] = (df["Overall"] >= 80).astype(int)

# Select numeric features only
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
X = df[numeric_cols].drop(columns=["Overall", "ID", "Unnamed: 0"], errors="ignore")
y = df["Above_Avg"]

# Handle missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

📊 Results
Accuracy: 0.9695

Classification Report:
               precision    recall  f1-score   support

           0       0.97      1.00      0.98      3531
           1       0.00      0.00      0.00       111

    accuracy                           0.97      3642
   macro avg       0.48      0.50      0.49      3642
weighted avg       0.94      0.97      0.95      3642

⚠️ Observations

Accuracy (97%) is very high, but this is misleading because the dataset is imbalanced (very few players have Overall ≥ 80).

The model fails to predict class 1 (Above Average players):

Precision, Recall, and F1-score are 0.0 for class 1.

This happens because SVM is biased toward predicting the majority class (0).

🔧 Possible Improvements

Balance the dataset using:

Oversampling (SMOTE)

Undersampling

Try different algorithms (e.g., Random Forest, Gradient Boosting, Logistic Regression).

Tune hyperparameters of SVM (C, gamma, kernel).

Normalize/scale features (SVM performs better with scaled data).

✅ Conclusion

The SVM classifier achieved 97% accuracy, but it did not correctly identify above-average players due to class imbalance.

Future improvements should focus on balancing the dataset and tuning hyperparameters to improve prediction performance for minority class players.
