# DATA PREPARARTION
import pandas as pd
cal= pd.read_csv("weight-height.csv")
print(cal)
print(cal.shape)
print(cal.describe())
print(cal.isnull().sum())
cal['Gender']=cal['Gender'].replace(['Male', 'Female'], [0, 1]).astype(int)
cal=cal.assign(weight_kgs=cal.Weight * 0.453592, height_cm=cal.Height * 2.54)
cal.drop(['Weight', 'Height'], axis=1, inplace=True)
print(cal.head())
print(cal.dtypes)

#   MODEL BUILDING     &    MODEL TRAINING
x=cal.drop('Gender', axis=1)
y=cal['Gender']
from sklearn.linear_model import LogisticRegression
lo_model=LogisticRegression()
lo_model.fit(x,y)
print(lo_model.coef_)
print(lo_model.intercept_)
y_pred=lo_model.predict(x)

    # MODEL EVALUATION
from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_pred))

    # MODEL DEPLOYMENT
from pickle import dump
dump(lo_model, open('gender_int.pkl', 'wb'))

