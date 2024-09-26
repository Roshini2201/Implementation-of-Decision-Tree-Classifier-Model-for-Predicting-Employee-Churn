# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2.  Import pandas module and import the required data set.
3. Find the null values and count them.
4.  Count number of left values.
5. From sklearn import LabelEncoder to convert string values to numerical values.
6. From sklearn.model_selection import train_test_split.
7.  Assign the train dataset and test dataset.
8. From sklearn.tree import DecisionTreeClassifier.
9. Use criteria as entropy.
10. From sklearn import metrics.
11. Find the accuracy of our model and predict the require values.
12. End the program. 
## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ROSHINI S
RegisterNumber: 212223240142
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/user-attachments/assets/248c2228-032d-44c5-921e-8f9e7d579402)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
