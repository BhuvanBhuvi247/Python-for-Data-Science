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

**Step - 1 : Understanding and Cleaning Data**

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

- Data shows the country names and indicators of countries growth.


vasv
