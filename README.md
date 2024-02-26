# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:RICHARDSON A
RegisterNumber:212222233005

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/ 
```

## Output:

# df.head() & df.tail()
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/f7c2030e-b396-4100-8185-9645f06cefa0)

# Values of X:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/fed6b8b4-4ffd-4a1a-b04c-069e8076b1b2)

# Values of Y:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/dbd21d59-3d91-4d7e-bc0f-42ed33018b3d)

# Values of Y prediction:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/9b36d802-178f-4e26-8364-90a61e4d1b2d)

# Values of Y test:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/c9348485-a9bd-4b26-814d-83d806d997be)

# Training set graph:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/f7e51f8d-8aac-4b37-b2c6-0c08211e6dd0)

# Test set graph:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/e83fb8be-cd5b-4c56-a306-0fe6b68e0b07)

# Value of MSE,MAE & RMSE:
![image](https://github.com/Richard01072002/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/141472248/25363217-d0b3-42b4-b8ba-1c401a0120a8)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
