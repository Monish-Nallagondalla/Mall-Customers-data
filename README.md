# K-Means Clustering with Python

This project demonstrates how to use the K-Means clustering algorithm to analyze datasets, including Mall Customer data and the Digits dataset. The project also includes data preprocessing steps like standardization and visualizations like the Elbow Method for selecting the optimal number of clusters.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Mall Customers Dataset](#mall-customers-dataset)
  - [Digits Dataset](#digits-dataset)
- [Results](#results)
- [License](#license)

## Overview

The K-Means clustering algorithm is used for unsupervised learning to partition data into distinct groups based on similarity. This project explores:

1. The Elbow Method to find the optimal number of clusters.
2. Visualization of cluster centers and patterns in data.
3. Standardizing features for better clustering performance.

## Installation

To run this project, ensure you have Python 3.7+ installed and install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### Mall Customers Dataset

1. Load the `Mall_Customers_V1.0.csv` dataset.
2. Perform clustering on features like `Annual Income` and `Spending Score`.
3. Visualize the results using the Elbow Method and scatter plots.

Run the following script:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
mcdata = pd.read_csv('path/to/Mall_Customers_V1.0.csv')
x = mcdata.iloc[:, [3, 4]].values

# Elbow Method
wcss_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()
```

### Digits Dataset

1. Use the `load_digits` dataset from `sklearn.datasets`.
2. Perform clustering on image pixel data.
3. Visualize the cluster centers.

Run the following script:

```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

# Visualize cluster centers
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

plt.show()
```

## Results

- **Mall Customers Dataset**: 
  - Used the Elbow Method to identify the optimal number of clusters.
  - Visualized clusters in 2D based on income and spending score.

- **Digits Dataset**:
  - Applied K-Means clustering on digit images.
  - Visualized the cluster centers representing each digit.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
