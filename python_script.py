import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
report_df = pd.read_csv('WH Report_preprocessed.csv')
BM = report_df.year == 2019
report2019_df = report_df[BM]
report2019_df
report2019_df.plot.scatter(x='Perceptions_of_corruption',y='Life_Ladder')
plt.figure(figsize=(12,12))
plt.scatter(report2019_df.Life_Ladder, report2019_df.Perceptions_of_corruption)

for _, row in report2019_df.iterrows():
    plt.annotate(row.Name, (row.Life_Ladder, row.Perceptions_of_corruption))

plt.xlabel('Life_Ladder')
plt.ylabel('Perceptions_of_corruption')

plt.show()
### Clustering Example using a 3-dimensional dataset
plt.figure(figsize=(12,12))
plt.scatter(report2019_df.Life_Ladder, report2019_df.Perceptions_of_corruption, c=report2019_df.Generosity,cmap='RdYlGn')
for _, row in report2019_df.iterrows():
    plt.annotate(row.Name, (row.Life_Ladder,
                 row.Perceptions_of_corruption),
                 c='grey',alpha=0.3)
plt.xlabel('Life_Ladder')
plt.ylabel('Perceptions_of_corruption')
plt.show()

plt.figure(figsize=(12,10))
plt.scatter(report2019_df.Life_Ladder, report2019_df.Perceptions_of_corruption,c=report2019_df.Generosity,cmap='binary')

plt.xlabel('Life_Ladder')
plt.ylabel('Perceptions_of_corruption')

plt.show()
## K-Means Algorithm

### Using K-Means to cluster a 2-dimensional dataset
from sklearn.cluster import KMeans
dimensions = ['Life_Ladder','Perceptions_of_corruption']
Xs = report2019_df[dimensions]
kmeans = KMeans(n_clusters=6)
kmeans.fit(Xs)

for i in range(6):
    BM = kmeans.labels_==i
    print('Cluster {}: {}'.format(i,report2019_df[BM].Name.values))
from sklearn.cluster import KMeans
dimensions = ['Life_Ladder','Perceptions_of_corruption']
Xs = report2019_df[dimensions]
kmeans = KMeans(n_clusters=6)
kmeans.fit(Xs)

for i in range(6):
    BM = kmeans.labels_==i
    print('Cluster {}: {}'.format(i,report2019_df[BM].Name.values))
plt.figure(figsize=(21,4))
plt.scatter(report2019_df.Life_Ladder, report2019_df.Perceptions_of_corruption)
for _, row in report2019_df.iterrows():
    plt.annotate(row.Name, (row.Life_Ladder, row.Perceptions_of_corruption),rotation=90)
plt.xlim([2.3,7.8])
plt.xlabel('Life_Ladder')
plt.ylabel('Perceptions_of_corruption')
plt.show()
dimensions = ['Life_Ladder','Perceptions_of_corruption']
Xs = report2019_df[dimensions]
Xs = (Xs - Xs.min())/(Xs.max()-Xs.min())
kmeans = KMeans(n_clusters=6)
kmeans.fit(Xs)

for i in range(6):
    BM = kmeans.labels_==i
    print('Cluster {}: {}'.format(i,report2019_df[BM].Name.values))
dimensions = [ 'Life_Ladder', 'Log_GDP_per_capita', 'Social_support',
              'Healthy_life_expectancy_at_birth', 'Freedom_to_make_life_choices',
              'Generosity', 'Perceptions_of_corruption', 'Positive_affect', 'Negative_affect']
Xs = report2019_df[dimensions]
Xs = (Xs - Xs.min())/(Xs.max()-Xs.min())
kmeans = KMeans(n_clusters=3)
kmeans.fit(Xs)

for i in range(3):
    BM = kmeans.labels_==i
    print('Cluster {}: {}'.format(i,report2019_df[BM].Name.values))
### Centroid Analysis
import seaborn as sns
clusters = ['Cluster {}'.format(i) for i in range(3)]

Centroids = pd.DataFrame(0.0, index =  clusters,
                        columns = Xs.columns)
for i,clst in enumerate(clusters):
    BM = kmeans.labels_==i
    Centroids.loc[clst] = Xs[BM].median(axis=0)

sns.heatmap(Centroids, linewidths=.5, annot=True, 
                    cmap='binary')
plt.show()