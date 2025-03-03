# House-Price-Prediction
The goal of this project is to predict house prices using machine learning techniques based on various features of the houses. The dataset provides a variety of features such as area, bedrooms, bathrooms, floors, year built, condition, and location, all of which influence the price of a house.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Uploading of Dataset
df = pd.read_csv('/content/House Price Prediction Dataset.csv')

# Seaborn Library is used for data visualization
sns.pairplot(df)

# To Read the dataset
df.head()

# To Drop One Coloumn
df=df.drop('Id',axis=1)

# To Deal with the Null Value 
df.isnull().sum()/df.shape[0]*100

# Pie Chart for Data Visualisation
plt.figure(figsize=(10,10))
plt.pie(df['Location'].value_counts(),labels=df['Location'].value_counts().index,autopct='%1.2f%%')

# Data Manipulation
df['yearsOld']=2024-df['YearBuilt']
df=df.drop('YearBuilt',axis=1)
df.head()

# Bar Chart for Data Visualisation
plt.figure(figsize=(10,10))
plt.bar(df['Condition'].value_counts().index,df['Condition'].value_counts())

# Some Measure Data Manipulations 
df['Condition'].value_counts()
df['Condition']=df['Condition'].map({'Poor':0,'Fair':1,'Good':2,'Excellent':3})
df.head()
df['Garage']=df['Garage'].map({'No':0,'Yes':1})
df.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Location']=le.fit_transform(df['Location'])
df.head()

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
df.head()

from sklearn.model_selection import train_test_split
x=df.drop('Price',axis=1)
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',input_dim=(x_train.shape[1])),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse',metrics=['mae'])

history=model.fit(x_train,y_train,epochs=100,validation_split=0.2)

model.evaluate(x_test,y_test)
