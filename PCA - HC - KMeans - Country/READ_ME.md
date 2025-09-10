# PCA - HC - KMeans - Country data

## Overview

The Country dataset contains details of indicators of a countries growth. As the data has many columns and rows, Principal Component Analysis and Clustering is done respectively to reduce the size of the data for better analysis. Hierarchical Clustering and KMeans Clustering both are being conducted for better results. Also a comparative study of Hierarchical Clustering and KMeans Clustering is done to find which mwthod is better for the data set. 

## Objectives
1) Importing the data set for analysis.
2) Cleaning the dataset for better analysis results.
3) Scaling data to have uniformity.
4) Perform Principal Component Analysis.
5) Perform Hierarchical Clustering.
6) Perform KMeans Clustering.
7) Compare the Clustering Results and Test both methods and select which method is better. 

## Procedure

**Step - 1 : Data Exploration and Cleaning Data**

```python
#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#reading CSV file
df = pd.read_csv('Country-data.csv')

df.head()
```
<img width="1069" height="272" alt="image" src="https://github.com/user-attachments/assets/8799576e-878d-4e13-8d55-0934363d4508" />

*Interpretation :*

- Data contains the country names and indicators of countries growth.

```python
#checking datatypes
df.dtypes
```
<img width="166" height="197" alt="image" src="https://github.com/user-attachments/assets/68a818ec-5434-4949-bc04-c3c6e30abffa" />

*Interpretation :*

- All the coulumns are in required data type. With 9 numeric columns and one nominal column.

 ```python
#checking null values
df.isnull().sum()
```
<img width="196" height="296" alt="image" src="https://github.com/user-attachments/assets/e9841385-232f-4b60-b2b8-82a587532861" />

*Interpretation :*

- The dataset has no null values, so no changes to be made.

 ```python
#checking duplicates
print ("Number of duplicate records : ",df.duplicated().sum())
```
<img width="374" height="40" alt="image" src="https://github.com/user-attachments/assets/a8c3ffb7-fa48-467e-8ed4-1619f556417b" />

*Interpretation :*

- The dataset has no duplicated records, so no changes to be made.

```python
#create duplicate df
dfa = df.copy()
#Create df with only numbers
dfn = df.select_dtypes(include='number')
dfn.describe()
```
<img width="1207" height="416" alt="image" src="https://github.com/user-attachments/assets/793cf6b6-1a68-4ff2-90af-00bbc60c5111" />

*Interpretation :*

- The basic description of the data shows that data is not scaled, so scaling has to be performed for further analysis.

```python
dfa.boxplot()
plt.title("Boxplot before Outlier Treatment")
plt.show()
```
<img width="868" height="650" alt="image" src="https://github.com/user-attachments/assets/e2bafe75-0920-4ece-a823-7936ec77ed26" />

*Interpretation :*

- The boxplot shows that there are too many outliers in income and gdpp columns. They have to be removed.

```python
#removing outliers
def remove_outlier(col):
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

for i in dfn:
    LL, UL = remove_outlier(dfn[i])
    dfn[i] = np.where(dfn[i] > UL, UL, dfn[i])
    dfn[i] = np.where(dfn[i] < LL, LL, dfn[i])

dfn.boxplot(figsize=(6,4))
plt.title("Boxplot after Outlier Treatment")
plt.show()
```
<img width="810" height="559" alt="image" src="https://github.com/user-attachments/assets/454f9a1c-b96e-4fd6-823e-1a0ff6fdd16f" />

*Interpretation :*

- The outliers have been removed and data is cleaned.

```python
#checking correlation using heatmap
plt.figure(figsize=(5,5))
sns.heatmap(dfn.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
<img width="759" height="767" alt="image" src="https://github.com/user-attachments/assets/aecfb1ce-f2ce-4036-9559-e24bf292069a" />

*Interpretation :*

- The Correlation Heatmap shows.
- Negative correlation in the following columns
- 1) child_mort
  2) income 
  3) life_exp
  4) gdpp
- Positive correlation in the following columns
- 1) child_mort
  2) exports
  3) income
  4) life_exp
- These are the columns with high correlation and affect the data to larger extent.
  
**Step - 2 : Scaling Data**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(dfn),columns=dfn.columns)
```
*Interpretation :*

- The data was distributed in different scales, soo all the data is is scaled to a uniform scale for better analysis.

**Step - 3 : Principal Component Analysis**

```python
from sklearn.decomposition import PCA

#Defining number of PCs to Generate
n=scaled_df.shape[1]

#Finding PCs for data
pca = PCA(n_components=n, random_state=1)
data_pca1 = pd.DataFrame(pca.fit_transform(scaled_df))

#The percentage of variance explained by each PC
exp_var = pca.explained_variance_ratio_

scaled_df.head()
```
<img width="1000" height="271" alt="image" src="https://github.com/user-attachments/assets/eff4bc36-bf02-4722-bf88-20a4b433f4b7" />

*Interpretation :*

- The components have been generated and checking for principal components.
- Components have been fit to the dataframe.
- Explained variance of each components has been calculated to check the components with high variance.

```python
#Visualise Explained Variance by individual components
plt.figure(figsize=(5,10))
plt.plot(range(1,len(exp_var)+1),exp_var.cumsum(),marker='o',linestyle='--')
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.show()
```
<img width="458" height="853" alt="download" src="https://github.com/user-attachments/assets/47ffc424-159e-4d45-b6c9-1c77303e5e06" />

*Interpretation :*

- The curve shows the cumulative sum of the explained variance of components
- Keeping the standard error of 5%, 5 components lie in the confidence range of 95%. So 5 Components have to be finalized.

```python
#Extracting Principle Components 
pc_comps = ['PC1','PC2','PC3','PC4','PC5']
data_pca = pd.DataFrame(np.round(pca.components_[:5,],2),index=pc_comps,columns=scaled_df.columns)
data_pca.T
```

<img width="439" height="451" alt="image" src="https://github.com/user-attachments/assets/3969c364-a25c-4aee-a631-bbde1435a3c2" />

*Interpretation :*

- The 5 Principal Components have been extracted and displayed 

**Step - 4 : Hierarchical Clustering**

```python
from scipy.cluster.hierarchy import dendrogram, linkage

wardlink = linkage(scaled_df, method = 'ward')
plt.figure(figsize=(14, 6))
dend = dendrogram(wardlink)

plt.title("Hierarchical Clustering Dendrogram")
plt.show()
```
<img width="1132" height="525" alt="download" src="https://github.com/user-attachments/assets/ba427504-30cb-41ba-85df-eeb1cc59a94a" />

*Interpretation :*

- L

```python
from scipy.cluster.hierarchy import fcluster

cluster = fcluster(wardlink,2,criterion='maxclust')

dfa['cluster_no'] = cluster
dfa.head()
```
<img width="1193" height="266" alt="image" src="https://github.com/user-attachments/assets/81975b1d-6b52-41d5-bf83-9fe475424d35" />

*Interpretation :*

- K

**Step - 5 : KMeans Clustering**

```python
from sklearn.cluster import KMeans

#checking how many clusters to be set
wss=[]
for i in range(1, 11):
    KM = KMeans(n_clusters=i)  
    KM.fit(scaled_df)
    wss.append(KM.inertia_)

plt.plot(range(1, 11), wss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WSS')
plt.title('Elbow Method')
plt.show()
```
<img width="580" height="453" alt="download" src="https://github.com/user-attachments/assets/0bb67cf7-e424-4a91-ad87-97b2c0dde6a9" />

*Interpretation :*

- K

```python
#assigning clusters
k_means = KMeans(n_clusters=2,random_state=12,n_init=10)
k_means.fit(scaled_df)
labels = k_means.labels_

dfa["clus_kmeans"] = labels
dfa.head()
```
<img width="1321" height="277" alt="image" src="https://github.com/user-attachments/assets/7a7c764c-1d2e-44f7-b6eb-b1b5e0049b09" />

*Interpretation :*

- L

```python
from sklearn.metrics import silhouette_samples,silhouette_score, calinski_harabasz_score, davies_bouldin_score

print ("Silhouette max value",silhouette_samples(scaled_df,labels).max())
```
<img width="446" height="28" alt="image" src="https://github.com/user-attachments/assets/237f16a8-2810-49ac-a687-c657e95cac78" />
```python
print ("Silhouette min value",silhouette_samples(scaled_df,labels).min())
```
<img width="454" height="33" alt="image" src="https://github.com/user-attachments/assets/34117d41-bca7-459e-a184-2d80e6f35236" />

*Interpretation :*

- L

**Step - 6 : Comparative Study**

```python
print("Silhouette Score of Hierarchical Clustering = ",silhouette_score(scaled_df,cluster))
print("Calinski-Harabasz Index:", calinski_harabasz_score(scaled_df,cluster))
print("Davies-Bouldin Index:", davies_bouldin_score(scaled_df,cluster))

cluster_counts = dfa.groupby("cluster_no")['country'].count()
print("\nCountries per Cluster:\n", cluster_counts)
```
<img width="739" height="248" alt="image" src="https://github.com/user-attachments/assets/dd8d7226-b9ca-400c-b6a1-a11967081477" />
```python
print("Silhouette Score of KMeans Clustering = ",silhouette_score(scaled_df,labels))
print("Calinski-Harabasz Index:", calinski_harabasz_score(scaled_df,labels))
print("Davies-Bouldin Index:", davies_bouldin_score(scaled_df,labels))

cluster_counts = dfa.groupby("clus_kmeans")['country'].count()
print("\nCountries per Cluster:\n", cluster_counts)
```
<img width="660" height="242" alt="image" src="https://github.com/user-attachments/assets/c64da54c-d53b-40dc-90a6-a2bc06893263" />

*Interpretation :*

- 
  
