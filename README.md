# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
Program to implement the SVM For Spam Mail Detection.
Developed by: JAISREE N
RegisterNumber: 212224060104
```
```
# Exp 11 – SVM for Spam Mail Detection with Visualization

# 1. Import Required Libraries
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Detect File Encoding
file_path = "spam.csv"   # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

# 3. Load Dataset
data = pd.read_csv(file_path, encoding=result['encoding'])

# 4. Basic Data Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# 5. Visualization: Spam vs Ham Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 6. Message Length Visualization
data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 7. Feature and Target Selection
x = data['v2'].values     # messages
y = data['v1'].values     # labels

# 8. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 9. Text Vectorization (Bag of Words)
cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# 10. Initialize and Train SVM
svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

# 11. Prediction
y_pred = svc.predict(x_test_vec)

# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 13. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

```

## Output:


<img width="904" height="659" alt="image" src="https://github.com/user-attachments/assets/53be681b-3211-4763-b872-235312875950" />
<img width="697" height="849" alt="image" src="https://github.com/user-attachments/assets/df50134b-335d-4dc1-a38f-dd5d061045ce" />
<img width="623" height="689" alt="image" src="https://github.com/user-attachments/assets/33fe409a-37c0-4092-a8bc-6b205965fba4" />






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
