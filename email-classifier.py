"""
Email Spam/Fraud/Genuine Classifier (Single File)
Uses: pandas, scikit-learn, joblib
Trains a model and predicts on new emails in one go.
"""

import warnings
from sklearn.exceptions import InconsistentVersionWarning
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress InconsistentVersionWarning (safe if you trust the model)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
print("📂 Loading email.csv...")
df = pd.read_csv('email.csv')

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract time features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # Mon=0, Sun=6
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Drop original datetime
df.drop('datetime', axis=1, inplace=True)

# Encode labels
le_subject = LabelEncoder()
le_category = LabelEncoder()
le_label = LabelEncoder()

df['subject_encoded'] = le_subject.fit_transform(df['subject'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['label_encoded'] = le_label.fit_transform(df['label'])

# Define features and target
feature_cols = ['subject_encoded', 'category_encoded', 'first_time', 'hour', 'day_of_week', 'is_weekend']
X = df[feature_cols]
y = df['label_encoded']

# -------------------------------
# 2. Train the Model
# -------------------------------
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained! Accuracy: {acc:.3f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_label.classes_, yticklabels=le_label.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save model and encoders
print("\n💾 Saving model and encoders...")
joblib.dump(model, 'model.pkl')
joblib.dump(le_subject, 'le_subject.pkl')
joblib.dump(le_category, 'le_category.pkl')
joblib.dump(le_label, 'le_label.pkl')
print("✅ Saved: model.pkl, le_subject.pkl, le_category.pkl, le_label.pkl\n")


# -------------------------------
# 3. Prediction Function
# -------------------------------
def predict_email(subject, datetime_str, category, first_time):
    """
    Predict if an email is spam, fraud, or normal.
    """
    # Load model and encoders
    model = joblib.load('model.pkl')
    le_subject = joblib.load('le_subject.pkl')
    le_category = joblib.load('le_category.pkl')
    le_label = joblib.load('le_label.pkl')

    # Parse datetime
    dt = pd.to_datetime(datetime_str)
    hour = dt.hour
    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    # Handle unseen subject/category
    try:
        subj_enc = le_subject.transform([subject])[0]
    except ValueError:
        print(f"[⚠️] Subject '{subject}' not seen before. Defaulting to 0.")
        subj_enc = 0

    try:
        cat_enc = le_category.transform([category])[0]
    except ValueError:
        print(f"[⚠️] Category '{category}' not seen before. Defaulting to 0.")
        cat_enc = 0

    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([[
        subj_enc, cat_enc, first_time, hour, day_of_week, is_weekend
    ]], columns=feature_cols)

    # Predict
    pred_encoded = model.predict(input_df)[0]
    pred_label = le_label.inverse_transform([pred_encoded])[0]

    return pred_label


# -------------------------------
# 4. Example Predictions
# -------------------------------
print("🧪 Running example predictions...\n")

examples = [
    {
        "subject": "Claim your free prize",
        "datetime_str": "2025-08-25 02:30:00",
        "category": "promotion",
        "first_time": 1
    },
    {
        "subject": "Weekly report due",
        "datetime_str": "2025-08-15 10:00:00",
        "category": "work",
        "first_time": 0
    },
    {
        "subject": "Your account is locked",
        "datetime_str": "2025-08-17 09:00:00",
        "category": "security",
        "first_time": 1
    }
]

for i, ex in enumerate(examples, 1):
    result = predict_email(**ex)
    print(f"📌 Example {i}: {result.upper()}")
    print(f"   Subject: '{ex['subject']}' → {result}\n")