import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

adv = pd.read_csv("/content/advertising.csv")

adv.head()

adv.info()

adv.shape

adv.describe()

#Data Cleaning
sns.heatmap(adv.isnull(),cbar=False,yticklabels=False)

#Outlier analysis
sns.boxplot(adv["TV"])

sns.boxplot(adv["Radio"])

sns.boxplot(adv["Newspaper"])

sns.boxplot(adv["Sales"])

sns.pairplot(adv)

"""DataFrame.corr

Compute pairwise correlation between columns.
"""

sns.heatmap(adv.corr(),annot=True)

#SIMPLE LINEAR REGRESSION
X_1=adv[["TV"]]
Y_1=adv["Sales"]

from sklearn.model_selection import train_test_split
X_train_1,X_test_1,Y_train_1,Y_test_1 = train_test_split(X_1,Y_1,test_size=0.4,random_state=1)

from sklearn.linear_model import LinearRegression

"""minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. 
Parameters fit_interceptbool, 
default=True. Whether to calculate the intercept for this model
"""

slm = LinearRegression(fit_intercept = True)

slm.fit(X_train_1,Y_train_1)

"""The coef_ contain the coefficients for the prediction of each of the targets. """

slm.coef_

slm.intercept_

y_1_pred = slm.predict(X_test_1)
y_1_pred

"""# Plot data and a linear regression model fit."""

sns.regplot(Y_test_1, y_1_pred, line_kws={'color':'red'}, ci=None)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

R2 = r2_score(Y_test_1,y_1_pred )
Mse = mean_squared_error(Y_test_1, y_1_pred)
Rmse = np.sqrt(Mse)
Mae = mean_absolute_error(Y_test_1, y_1_pred)

R2

Mse

Rmse

Mae

"""root-mean-square deviation"""

res=(y_1_pred-Y_test_1)
res

sns.distplot(res, bins = 15)

adv

#value check
pred_y_table = pd.DataFrame({"Actual Value":Y_test_1,"Predicted Value":y_1_pred,"Difference":res})
pred_y_table

pred_y_table.describe()

#MULTIPLE LINEAR REGRESSION
X=adv[["TV","Radio","Newspaper"]]
Y=adv["Sales"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=1)

mlm = LinearRegression(fit_intercept = True)

mlm.fit(X_train,Y_train)

mlm.coef_

mlm.intercept_

y_pred = mlm.predict(X_test)

sns.regplot(Y_test, y_pred, line_kws={'color':'red'}, ci=None)

Y_test.shape

k=3
n=80

r2 = r2_score(Y_test,y_pred )
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
adj_r2= 1- ((1-r2)*(n-1)/(n-k-1))

r2

mse

rmse

mae

adj_r2

res_1=(Y_test-y_pred)

sns.distplot(res_1, bins = 15)

adv

#value check
pred_y_table_1 = pd.DataFrame({"Actual Value":Y_test,"Predicted Value":y_pred,"Difference":res_1})
pred_y_table_1

pred_y_table_1.describe()
