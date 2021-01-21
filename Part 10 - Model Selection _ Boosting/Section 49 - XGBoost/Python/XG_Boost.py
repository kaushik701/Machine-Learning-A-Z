#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from xgboost import XGBClassifier
# %%
data = pd.read_csv('Data.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)
# %%
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
# %%
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
# %%
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
print(accuracies.mean()*100)
print(accuracies.std()*100)
# %%
