# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection and Preprocessing

Load the dataset using pandas from the given URL.

Remove unnecessary columns (CarName, car_ID).

Convert categorical variables into numerical form using one-hot encoding.

Step 2: Data Splitting

Separate the dataset into:

Independent variables (X)

Dependent variable (y → price)

Split the data into training and testing sets (80% training, 20% testing).

Step 3: Model Training and Evaluation

Create a Linear Regression model using scikit-learn.

Train the model using the training dataset.

Perform 5-fold cross-validation to evaluate model performance.

Print the intercept, coefficients, and mean cross-validation score.

Step 4: Prediction and Visualization

Predict car prices using the test dataset.

Plot a scatter graph of Actual Prices vs Predicted Prices.

Draw a reference line to show perfect prediction.

## Program:
```

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")


data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)


X = data.drop('price', axis=1)
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()



model.fit(X_train, y_train)


cv_scores = cross_val_score(model, X, y, cv=5)


print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())


print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


predictions = model.predict(X_test)


plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
```

## Output:
<img width="935" height="272" alt="image" src="https://github.com/user-attachments/assets/03e3638c-3d31-452f-b147-57f63c202553" />

<img width="1267" height="587" alt="image" src="https://github.com/user-attachments/assets/8831deb3-f6ab-45af-afa8-e1b8ae0f993a" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
