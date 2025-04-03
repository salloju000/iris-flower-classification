import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load dataset and preprocess
sns.get_dataset_names()
data = sns.load_dataset("iris")

# Encode the target variable
le = LabelEncoder()
data['new_species'] = le.fit_transform(data['species'])

# Features and targets
x = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values

# Scaling the input features
scale = StandardScaler()
x_scaled = scale.fit_transform(x)

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, train_size=0.80, random_state=3)

# Model building and training
model = KNeighborsClassifier(n_neighbors=8, p=2)
model.fit(xtrain, ytrain)

# Model prediction
ypred = model.predict(xtest)

# Evaluate the model
cm = confusion_matrix(ytest, ypred)
acc = accuracy_score(ytest, ypred)

# Save the model for later use in the UI
joblib.dump(model, "sid_model.pkl")

# Optionally, return the accuracy and confusion matrix for future use


def get_model_info():
    return acc, cm
