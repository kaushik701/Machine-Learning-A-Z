#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,[3,4]].values
# %%
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,init='k-means++',random_state=43)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()
# %%
Kmeans = KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans = Kmeans.fit_predict(X)
# %%
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100,c='red',label='Cluster-1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100,c='blue',label='Cluster-2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100,c='green',label='Cluster-3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100,c='yellow',label='Cluster-4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100,c='pink',label='Cluster-5')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300,c='orange',label='Centroids')
plt.title('Clustering of Customers')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Income (1-100)')
plt.legend()
plt.show()
# %%
