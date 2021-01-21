#%%
# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 0)

# train the dataset in linear regression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict the result using the X_test dataset
y_pred = regressor.predict(X_test)

# visualizing the graph with training dataset
plt.scatter(X_train,y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the graph with test dataset
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# %%
