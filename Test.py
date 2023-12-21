from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plot

File_path = 'D:/data/'
File_name = 'car_data'

df = pd.read_excel(File_path+File_name)

#Preprocess
df.dropna(columns=['User ID'], inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['AnnualSalary'].fillna(df['AnnualSalary'].mean(), inplace=True)
enconders = []
for i in range(0, len(df.columns)-1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
    encoders.append(enc);

x = df.iloc[:, 0:5]
y = df['test']
x_train, x_test, y_train, y_test = train_test_split(x,y)

x_pred = ['Male','35','20000','No']
for i in range(0, len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
x_pred_adj = np.array(x_pred).reshape(-1,5)
x_pred = model.predict(x_pred_adj)
print('Prediction x: ', x_pred[0])
y_pred = [Female,47,33500,Yes]
for i in range(0, len(df.columns)-1):
    y_pred[i] = encoders[i].transform([y_pred[i]])
y_pred_adj = np.array(y_pred).reshape(-1,5)
y_pred = model.predict(y_pred_adj)
print('Prediction y: ', x_pred[0])
score = model.score(x,y)
print('Accuracy: ','{:.2f}'.format(score))