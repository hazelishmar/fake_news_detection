import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load datasets
df_true = pd.read_csv('True.csv')
df_false = pd.read_csv('Fake.csv')

# Add labels: 1 for true news, 0 for fake
df_true['label'] = 1
df_false['label'] = 0

# Combine and shuffle
df = pd.concat([df_true, df_false], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Use 'text' column; change if needed
if 'text' in df.columns:
    texts = df['text']
elif 'content' in df.columns:
    texts = df['content']
else:
    raise Exception("Your CSV files must contain a 'text' or 'content' column.")

labels = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
