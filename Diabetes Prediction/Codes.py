
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file

import pandas as pd
df = pd.read_csv("/content/diabetes.csv")
df.head(3)

#unique features in the targeted variable
df.Outcome.unique()

# Commented out IPython magic to ensure Python compatibility.
#checking if there is any null value or not by using heatmap
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

sns.heatmap(df.isnull(),yticklabels=False)

sns.set_style("whitegrid")
sns.countplot(df.Outcome, data=df)

plt.scatter(df.Age, df.BloodPressure)
plt.xlabel('Age')
plt.ylabel('BloodPressure')

plt.scatter(df.Age, df.Pregnancies)
plt.xlabel('Age')
plt.ylabel('Pregnancies')

plt.scatter(df.Age, df.BMI)
plt.xlabel('Age')
plt.ylabel('BMI')

col=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
for i in col:
    sns.boxplot(df[i])
    plt.show()

x=df.drop(['Outcome'], axis='columns')
y=df.Outcome

print('The independent variabl')
print(x.head(3))
print('The dependent variable')
print(y.head(3))

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(x))
print(z)

df1=pd.DataFrame(data=z,columns=x.columns )
df1.head(10)

#checking for the outlier by using IQR (Interquartile range)
#The IQR describes the middle 50% of values when ordered from lowest to highest
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

A = ((x < (Q1 - 1.5 * IQR)) |(x > (Q3 + 1.5 * IQR)))
print(A)

#removing all the outlier from the dataframe
X_remove_outlier = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
X_remove_outlier.shape

sns.set_style("whitegrid")
sns.countplot(X_remove_outlier.Outcome, data=X_remove_outlier)

X=X_remove_outlier.drop(['Outcome'], axis='columns')
Y=X_remove_outlier.Outcome
print('The independent variable')
print(X.head(10))
print('The target part')
print(Y.head(10))

# MinMaxScaler preserves the shape of the original distributio
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
fetures=scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

print(len(X_train))
print(len(X_test))

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

# Compute confusion matrix to evaluate the accuracy of a classification.
from sklearn.metrics import confusion_matrix
y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
cm

model.score(X_test,y_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
model_RF=RandomForestClassifier(n_estimators=100, min_samples_leaf=100, max_features=5)
model_RF.fit(X_train,y_train)

y_pred_RF=model.predict(X_test)
cm_RF=confusion_matrix(y_test,y_pred)
cm_RF

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_RF))

from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)

y_pred_LR=model.predict(X_test)
cm_LR=confusion_matrix(y_test,y_pred)
cm_LR

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_LR))

from sklearn.svm import SVC
model_SVC=SVC(kernel='linear')
model_SVC.fit(X_train,y_train)

y_pred_SVC=model.predict(X_test)
cm_SVC=confusion_matrix(y_test,y_pred)
cm_SVC

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_SVC))

sns.heatmap(cm_SVC,annot=True)
plt.xlabel('True')
plt.ylabel('Predicted')
