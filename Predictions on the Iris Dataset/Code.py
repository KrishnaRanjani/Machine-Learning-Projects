import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Iris = pd.read_csv("/content/Iris.csv")

Iris.head()

Iris.info()

Iris.describe()

#Data is clean
#Exploratory Data Analysis With Iris
plt.figure(figsize = (15,8))
sns.set(style="darkgrid")
sns.scatterplot(data=Iris,x="SepalLengthCm",y='SepalWidthCm',hue="Species")

plt.figure(figsize = (15,8))
sns.set(style="darkgrid")
sns.scatterplot(data=Iris,x="SepalLengthCm",y='SepalWidthCm',hue="Species")

#This plot clearly shows that Iris-setosa are very easy to identify whereas the other 2 are mostly similar in nature

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=Iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=Iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=Iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=Iris)

#Train_Test_split
X=Iris[["PetalLengthCm","PetalWidthCm","SepalLengthCm","SepalWidthCm"]]
Y=Iris["Species"]

#Data Preprocessing
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=500)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#KNN
#Training model
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors =5, metric="minkowski",p=2)
classifier.fit(X_train,Y_train)

#Testing model
y_pred = classifier.predict(X_test)
y_pred

Y_test

#Model_evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(Y_test,y_pred)

print(accuracy_score(Y_test, classifier.predict(X_test)))

sns.heatmap(cm,annot=True)

print(classification_report(Y_test,y_pred))

"""Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.

F1 Score is the weighted average of Precision and Recall. 

Support is the number of actual occurrences of the class in the specified dataset.

"""

#Logistic Regression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression()
classifier_1.fit(X_train,Y_train)

y_pred_1 = classifier_1.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Y_test, classifier_1.predict(X_test)))

CM=confusion_matrix(Y_test, y_pred_1)

sns.heatmap(CM,annot=True)

CR=classification_report(Y_test, y_pred_1)

print(CR)
