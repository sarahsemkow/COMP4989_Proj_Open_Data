import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

file_path = "../data/200_angles/random_sample_200_all.csv"
# dataframe of all the data
data = pd.read_csv(file_path)

# Shuffle the DataFrame
shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = shuffled_data.drop("pose", axis=1)
y = shuffled_data["pose"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
predictions = classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"RFC Accuracy: {accuracy}")

print("RFC Classification Report:")
print(classification_report(y_test, predictions))

# result_df = pd.DataFrame({'Prediction': predictions, 'Actual': y_test})
# result_df.to_csv('randomforestclassifier.generated_csv', index=False)


'''
Accuracy: 0.9671361502347418
Classification Report:
              precision    recall  f1-score   support

     downdog       1.00      0.97      0.99       119
     goddess       0.97      0.89      0.93        37
       plank       0.93      0.98      0.95        52
        tree       1.00      0.97      0.98        88
     warrior       0.93      0.98      0.95       130

    accuracy                           0.97       426
   macro avg       0.97      0.96      0.96       426
weighted avg       0.97      0.97      0.97       426


Process finished with exit code 0

'''