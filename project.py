import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import scipy.cluster.hierarchy as sch
import altair as alt 
%matplotlib inline
%matplotlib notebook 
df = pd.read_csv('customer_segmentation.csv')



#
dummies_df = pd.get_dummies(df, columns = ['order_status','payment_type', 'customer_city', 'customer_state', 'seller_city','seller_state', 'product_category_name','product_category_name_english'])

dummies_df.head()



#
frequencies = df.groupby(
    by=['customer_id'], as_index=False)['order_purchase_timestamp'].count()
frequencies.columns = ['Frequencies Customer ID', 'Frequency']
frequencies.head()


#
df['Orders_value'] = df['payment_value']
monetary = df.groupby(by='customer_id', as_index=False)['Orders_value'].sum()
monetary.columns = [' Monetary Customer ID', 'Monetary value']
monetary.head()

#
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

recency = df.groupby(by='customer_id',
                        as_index=False)['order_purchase_timestamp'].max()

recency.columns = ['Customer ID', 'Latest Purchase']

recent_date = recency['Latest Purchase'].max()

recency['Recency'] = recency['Latest Purchase'].apply(
    lambda x: (recent_date - x).days)
    
recency.head()




#
rfm_dataset = None

rfm_dataset = pd.DataFrame(rfm_dataset)

rfm_dataset = pd.concat([recency,monetary,frequencies], axis = 1)

cols = [3,5]

rfm_dataset = rfm_dataset.drop(rfm_dataset.columns[cols],axis = 1)



#scaling the data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = MinMaxScaler()
sc_features = rfm_dataset.copy() 



new = sc.fit_transform(sc_features['Monetary value'].array.reshape(-1,1))
new2 = sc.fit_transform(sc_features['Recency'].array.reshape(-1,1))
#new3 = sc.fit_transform(sc_features['Frequency'].array.reshape(-1,1))
sc_features['Monetary value'] = new
sc_features['Recency'] = new2 
#sc_features['Frequency'] = new3
sc_features.head()




#
from sklearn.cluster import KMeans
features = ['Recency','Monetary value','Frequency']
X = sc_features[features]


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42, n_init = 3, algorithm = 'lloyd')
y_kmeans = kmeans.fit_predict(X)

X = X.values 


# Visualising the clusters
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],X[y_kmeans == 0,2], s = 40, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],X[y_kmeans == 1,2], s = 40, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],X[y_kmeans == 2,2], s = 40, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1],X[y_kmeans == 3,2], s = 40, c = 'cyan', label = 'Cluster 4')
ax.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1],X[y_kmeans == 4,2], s = 40, c = 'magenta', label = 'Cluster 5')
#ax.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1],X[y_kmeans == 5,2], s = 40, c = 'black', label = 'Cluster 6')
#plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'orange', label = 'Cluster 7')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black', label = 'Centroids')  #as one can see centroids are pretty close..
plt.title('Clusters of customers')
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.legend()
plt.show()