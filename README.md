# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries `pandas`, `matplotlib.pyplot`, and `KMeans` from `sklearn.cluster`.  
2. Load the dataset `Mall_Customers.csv` using `pd.read_csv()` and preview the data using `.head()`.  
3. Use `.info()` to understand the structure and data types of the dataset.  
4. Check for missing values using `.isnull().sum()` to ensure data completeness.  
5. Initialize an empty list `wcss` to store the within-cluster sum of squares (WCSS).  
6. Use a `for` loop to iterate over numbers 1 to 10, creating a `KMeans` model with `n_clusters=i` and the `k-means++` initialization method.  
7. Fit the model to the dataset columns `Annual Income (k$)` and `Spending Score (1-100)` using `.iloc[:, 3:]`. Append the inertia (WCSS) to the `wcss` list.  
8. Plot WCSS values against the number of clusters using `plt.plot()`, labeling the axes and the plot title as "Elbow Method".  
9. Create a new `KMeans` model with `n_clusters=5` and fit it to the same dataset columns.  
10. Predict cluster labels for the dataset and assign them to a new column `cluster` in the original dataset.  
11. Split the dataset into subsets `df0`, `df1`, `df2`, `df3`, and `df4` based on the `cluster` column values.  
12. Plot scatter plots for each cluster using `plt.scatter()` with `Annual Income (k$)` on the x-axis and `Spending Score (1-100)` on the y-axis. Use distinct colors and labels for each cluster.  
13. Add a legend and title "Customer Segments" to the plot and display it.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Dhivya Dharshini B
RegisterNumber:  212223240031
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/admin/Desktop/INTR MACH/Mall_Customers.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss= [] #with-in the cluster sum of square

for i in range(1, 11):
    kmeans= KMeans(n_clusters = i , init= "k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11) , wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters= 5)
km.fit(data.iloc[:, 3:])

KMeans(n_clusters= 5)

y_pred= km.predict(data.iloc[:,3:])
y_pred

data["cluster"]= y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"] , c="red", label="cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"] , c="black", label="cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"] , c="blue", label="cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"] , c="green", label="cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"] , c="magenta", label="cluster4")
plt.legend()
plt.title("Customer Segements")
```
## Output:
### Elbow method
![image](https://github.com/user-attachments/assets/58fe6ec7-1168-4f7a-97cb-ae8341df32d0)

### Y- Prediction
![image](https://github.com/user-attachments/assets/c19298aa-d647-4178-a42d-c1f915580aad)

### Customer Segments(Cluster)
![image](https://github.com/user-attachments/assets/a85d1a33-a76a-4ea5-8411-7cf70a66d2bb)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
