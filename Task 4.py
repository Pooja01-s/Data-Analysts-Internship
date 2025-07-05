# Sentiment Analysis on Tweets using NLP
#Importing Required modules/libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
import string

#Loading datasets.
df = pd.read_csv("D:/CODETECH INTERN/Twitter_Data.csv")
df = df.dropna(subset=['clean_text', 'category'])
df['category'] = df['category'].astype(int)

# Basic Exploratory Data Aanalysis.
print("Category distribution:")
print(df['category'].value_counts())

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)      # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()     # Remove extra whitespace
    return text

# Apply preprocessing
df['clean_text'] = df['clean_text'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']

#Splitting the data as train and test data using train_test_split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Results or Insights
coefficients = model.coef_
feature_names = vectorizer.get_feature_names_out()
for idx, label in enumerate(['Negative', 'Neutral', 'Positive']):
    top_features = np.argsort(coefficients[idx])[-10:]
    print(f"\nTop words for {label} sentiment:")
    print([feature_names[i] for i in top_features])
