#Importing required modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#Loading Dataset
df=pd.read_csv("D:/CODETECH INTERN/Lung Cancer.csv")
print("Data View:/n",df.head())

# Drop non-informative or high-cardinality features
df.drop(columns=['id', 'diagnosis_date', 'end_treatment_date'], inplace=True)
# Handle missing values if any
df = df.dropna()

# Encode categorical variables
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for column in cat_cols:
    df[column] = le.fit_transform(df[column])

#Splitting the data as train and test data
X = df.drop("survived", axis=1)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training the Model using RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Evaluating the model
y_pred = clf.predict(X_test)

#Insights or Result
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100,2),"%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix with Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Purples", xticklabels=[0,1], yticklabels=[0,1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()