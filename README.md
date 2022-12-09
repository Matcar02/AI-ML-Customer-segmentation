# Projectrepo2
 Repository Ai project
<h1> Machine learning Project: Customer Segmentation</h1>
<h3> Group members: Carucci Matteo, Agudio Tommaso, Natoli Vittorio Alessandro </h3>
<h2>Introduction</h2>
<h3> In the first part of the project, we deal with a customers database where customers' orders in Brazil are registered. There are many information stored for each order, including the price spent and also some relevant information about the customer and sellers themselves; </h3>
<h3> We then have done some data exploration in order to check for null values, duplicates and useless features. </h3>
<h3> Afterward we have done the RFM analysis and calculated the Recency, Frequency and Monetary values, plotted the RFM dataset to have a better look at the data and noticed that customers who have purchased more recenlty spent more and had a higher frequency. To confirm our hypothesis we used the function corr() on the dataset and noticed that there was positive correlation between frequency and monetary value, thus confirming our intial observation. Afterwards we applied K-means and Hierarchial Clustering on the RFM and plotted the results in a 3D plot, while our third method was Spectral Clustering. Afterwards we did some further data exploration to see if we could consider other aspects of the dataset to potentially confirm or reasses the results obtained in the clustering. Lastly after finding some interesting features in the dataset, we used two different alternatives; Principal Component Analysis and Autoencoders ANN. When we gather our results, we applied the clustering methods and made our conclusions. </h3>

 [Section 2] Methods – Describe your proposed ideas (e.g., features, algorithm(s), training overview, design choices, etc.) and your environment so that: o A reader can understand why you made your design decisions and the reasons behind any other choice related to the project o A reader should be able to recreate your environment (e.g., conda list, conda envexport, etc.) o It may help to include a figure illustrating your ideas, e.g., a flowchart illustrating the steps in your machine learning system(s)

<h2> Methods Used</h2>
<h3>The main methods we used where KMeans, Hierarchical Cluster Clustering, Spectral Clustering, PCA and Autoencoders</h3>
<h3>The first step was to calculate the RFM, where as we have stated before, RFM stands for Recency (How long has been since the last purchase of a customer), Frequency (The frequency of purchases of a customer), and Monetary (How much a customer has spent over its time using the marketplace). To get these values, we simply used a groupby on the original dataframe (because we don't need any categorical features), applied some conditions, obtained the three different dataframes and concatenated them into a single one. An important aspect was that, after many trials we decided not to scale data with either MinMax or Linear scaling (Standard) since it would compromise the values of the frequency, with many equal to 0; moreover neither the clusters distance nor the silhouette score improve significantly. After having obtained this new dataframe we applied the following two algorithms.

<h3>KMeans and Hierarchical Cluster Clustering: 

The process for this two algorithms was very similar. We started from the RFM dataframe that we calculated previously and used respectively the Elbow Method, for KMeans, and a Dendogram, for Hierarchical Clustering, to then analyze the ideal number of clusters to obtain the best results. For both the methods we had a few hyperparameters to tune, when we used KMeans we first used GridSearch to better interpret and tune the parameters which where: the number of clusters, clustering algorithm (lloyd or elkan) and number of iterations for centroids. We obtained that the parameters to produce better results are likely elkan, five clusters and three iterations to find the centroids. An important aspect though is that after a few computations, we decided that four clusters is a good compromise. We then created the class that allows us to change the hypermarameters and then compute the resulting graph after applying the KMeans. Instead for HC clustering, after choosing the amount of centroids as the result of the Dendogram, we had two major hyperparameters to tune, the linkage (ward and average work similarly), that is, the metrics to use to determine the distance betweeen clusters, and the affinity, what type of distance measure to use to compute such linkage. For the sake of consistency, we decided to carry out the algorithm with four clusters. For both methos we implemented a 3D plot, using plotly to better visualize the data. Our last step was to compute the silhouette score, this is done to better understand how well the clusters were separated by basically taking the intra cluster distance into account. We did this for both our methods and obtained that, KMeans had a silhouette score of 0.0.816001, while HC 0.79044, slightly worse than KMeans.


Spectral Clustering:

We used this third method because we wanted to check if it had better results, epecially because it works better when there are a lower number of clusters. The way this method works is by creating an affinity matrix, where each datapoint is compared to others by assessing the "similarity", that is, sklearn builds a graph with datapoints as nodes, and uses the number of common neigbors(nearest) to identify specific communities. From the plot obtained, we notice that the cluster separation is fine, but the main issue is that the clustering identified too many high spenders and too few risk-customers, which is incongruous to the Exploratory Data Analysi, we previously carried out, which shows that only a few people spend a lot, while most spend the same amount.
We also computed the silhouette score for this method and obtained 0.34288 


Principal Component Analysis:

As we have stated before, after computing the clustering on the RFM, we decided to use PCA to detect some important features and preprocess the data and then compute the clustering on these new features. In particular, we analyzed the payment type and the number of installments, the customer state and lastly the product category. All of these features could have some interesting insights, for example the customer state tells us some important information on the demographics, which is an important aspect considering the huge disparity between the rich and poor in Brazil.

The first step was creating the dataframe with all the features we want to include in our analysis and scale the data, we then plot the cumulative plot of the PC that allows us to understand how much information n components provide together. Afterwards we apply the Elbow Method to decide the number of clusters considering the PCA components and we deduced that we should use four clusters again. The next step is to tune the  hyperparamenters and visualize the results of PCA, which is a way to detect the most important features in the new dataset. Lastly we visualized the clusters made by the KMeans after applying the PCA transformation and plot the data in 2D. The results are not really that different from those obtained previously. We have a few customers that are more likely to be top spends and other who are mid and low spenders.


Autoencoder ANN:

This is the last method we used, where the main goal was to reduce the dimensionality of the data using the autoencoder, a sort of Artifical Neural Network. This method works as the following:
    -The autoencoder learns a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore insignificant data (“noise”).

    -It has 2 main parts: an encoder that maps the message (data) to a code, and a decoder that reconstructs the message (processed data) from the code, that is the decoder extracts the most relevant patterns and information we want to retrieve. 

    The below image is a representation of the process.
    ![autoencoders.png](attachment:download.png)

The most important step was to apply the scaling on the data, we then applied the autoencoder, got the output data (obtained after doing the epochs and encoding) and use the KMeans algorithm. After ploting the data, we observe that we obtain different results from our previous analysis, this is because the Autoencoder ANN clusters the customers using the RFM and other characterstics obtained by the ANN.
</h3>

<h2>Experimental Design</h2>

<h3>The main way we tested the obtained results is through silhoutte score and the method describe() to deduce the information of each cluster. 

The silhoutte was used to measure and better understand how well the clusters were separated, by basically taking the intra cluster distance into account. We then tuned the hyperparameters to see how the silhoutte score changed, thus decided which was the optimal combination. To do so, we did a gridsearch which produced, as a result, the parameters to obtain likely the best score. This was used to the KMeans, while for HC and Spectral we simpy used to silhoutte score and manually changed the parameters.

The other way to validate the information was through the method describe, which we used for the PCA and Autoencoders, but also for the other three methods. We did a sort of deep dive into the indivual clusters obtained and checked some important insights and derived some information on the cluster.
</h3>

<h2> Results</h2>
<h3></h3>

