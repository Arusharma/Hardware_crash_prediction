import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,r2_score
from pyramid.arima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC ,SVR
from sklearn import linear_model,svm
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 

#from pyramid.arima import auto_arima

df=pd.read_csv("dataBigML.csv")
cols = df.columns
date=df['Date']
df.drop(['Date'], axis=1, inplace=True)
row1=df.index[df.Failure=='Yes']
df.loc[df.Failure=='Yes', 'Failure'] = 1
df.loc[df.Failure=='No', 'Failure'] = 0

labelencoder_df = LabelEncoder()
df.loc[:, 'Operator'] = labelencoder_df.fit_transform(df.loc[:, 'Operator'])
X=df.loc[:, df.columns != 'Failure']
y=df['Failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test) 

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

#row1=df.index[df.Failure=='1']
i=row1[2]
row_test=X.iloc[[i]]
temp=90
cpu=90
print(row_test)
row_test.iat[0, 0] = temp
row_test.iat[0, 1] = cpu
predict_y=clf.predict(row_test)
print("This is the value of the predicted",predict_y)
row_test['Failure']=predict_y
df['Hrs']=np.zeros(shape=(len(df),1))
#row1=df.index[df.Failure=='Yes']
start=0
for j in range(len(row1)):
    end=row1[j]
    val=df.loc[row1[j],'Hours Since Previous Failure']
    df.loc[start:end,"Hrs"] = val - df.loc[start:end,"Hours Since Previous Failure"]
    start=end

start=end
end = len(df) - 1
val=df.loc[len(df)-1,'Hours Since Previous Failure']
#print("last value",val)
df.loc[start:end,"Hrs"] = val - df.loc[start:end,"Hours Since Previous Failure"]
df.Hrs[df.Hrs < 0 ]= 0
X=df.loc[:, df.columns != 'Hrs']
y=df['Hrs']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
clf=BaggingRegressor()
clf.fit(X_train,y_train)  
y_pred=clf.predict(X_test)


print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))

y_extra=clf.predict(row_test)
print("The no of hrs before this thing craesfges is",y_extra)


# Make the plot

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(df['Temperature'], df['Humidity'], df['Hrs'], cmap=plt.cm.jet, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
plt.show() 
'''
 

plt.figure(0)
plt.plot(date[:len(y_test)], y_test)
#plt.figure(1)
plt.plot(date[:len(y_test)], y_pred,'r')
plt.savefig('graph.jpg')
plt.show()

