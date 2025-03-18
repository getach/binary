# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:09:08 2025

@author: getch
"""
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree  
import pandas as pd
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('mammographic_masses.data.txt', na_values = '?')
df.columns = ['BI_RADS', 'age', 'shape', 'margin', 'density','severity']
df = df.dropna()
df.reset_index(inplace=True)
scaler = StandardScaler()
x = df[['BI_RADS', 'age', 'shape', 'margin', 'density']]
y = df['severity']
x = scaler.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Decision Tree Classifier
# max_depth limits tree depth to prevent overfitting; random_state ensures reproducibility
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)

# Step 4: Train the model
dt_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report (precision, recall, f1-score for each class)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Step 7: Visualize the Decision Tree (optional)
