import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# =========================
# 1. LOAD DATASETS
# =========================
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Separate features and labels
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]

X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# =========================
# 2. ENCODE TARGET LABELS
# =========================
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# =========================
# 3. DEFINE REGULARIZED MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,              # controls overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# =========================
# 4. CROSS-VALIDATION
# =========================
cv_scores = cross_val_score(
    model,
    X_train,
    y_train_enc,
    cv=5,
    scoring="accuracy"
)

print("Cross-Validation Accuracy:")
print("Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# =========================
# 5. TRAIN FINAL MODEL
# =========================
model.fit(X_train, y_train_enc)

# =========================
# 6. TEST SET EVALUATION
# =========================
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test_enc, y_pred)

print("\nTest Set Accuracy:", test_accuracy)

# =========================
# 7. SAVE MODEL & ENCODER
# =========================
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel and label encoder saved successfully.")
