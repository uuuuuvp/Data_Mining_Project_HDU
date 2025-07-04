import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os
os.environ['QT_DEBUG_PLUGINS'] = '0'

# Step 1: 数据清洗
raw_data = pd.read_csv("/mnt/e/Desktop/Online_RetailDataset_Mining-master/Online_RetailDataset_Mining-master/online_retail.csv", encoding='ISO-8859-1')
clean_data = raw_data[raw_data['CustomerID'].notnull()]
clean_data.to_csv("raw_data.csv", index=False)

# Step 2: Frequency-Monetary模型
cluster_data = clean_data.drop(columns=['StockCode', 'Description', 'Country'])
cluster_data['Monetary'] = cluster_data['Quantity'] * cluster_data['UnitPrice']
cluster_data['Date'] = cluster_data['InvoiceDate'].astype(str).str.split().str[0]

monetary_data = cluster_data.groupby(['CustomerID', 'InvoiceNo', 'Date'])['Monetary'].sum().reset_index()
current_date = pd.to_datetime('2011-01-21')
monetary_data['Date'] = pd.to_datetime(monetary_data['Date'], errors='coerce')
monetary_data['Recency'] = (current_date - monetary_data['Date']).dt.days
monetary_data['Frequency'] = 1

FM_data = monetary_data.groupby('CustomerID').agg({'Monetary': 'sum', 'Frequency': 'sum'}).reset_index()
FM_data['Monetary'] = FM_data['Monetary'].apply(lambda x: max(x, 0))

# Step 3: RFM模型
recency = monetary_data.groupby('CustomerID')['Recency'].min().reset_index()
RFM_data = pd.merge(FM_data, recency, on='CustomerID')

# Step 4: 异常值处理
def outlier_IQR(series, multiple=1.5, replace=True, revalue=None):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + multiple * iqr
    outliers = series > upper_bound
    if replace:
        series[outliers] = revalue
    return series

RFM_data['Monetary'] = outlier_IQR(RFM_data['Monetary'], revalue=1147)
RFM_data['Frequency'] = outlier_IQR(RFM_data['Frequency'], revalue=4)

RFM_data['Recency'] = -RFM_data['Recency']
RFM_data['Frequency'] = np.log1p(RFM_data['Frequency'])

scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))

RFM_data_normalized = RFM_data.copy()
RFM_data_normalized[['Monetary', 'Frequency', 'Recency']] = scaler.fit_transform(RFM_data[['Monetary', 'Frequency', 'Recency']])

print("鲁棒归一化后数据描述：")
print(RFM_data_normalized[['Monetary', 'Frequency', 'Recency']].describe())

# Step 6: 确定最佳k值
k_list = range(1, 10)
sse = []

for k in k_list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(RFM_data_normalized[['Monetary', 'Frequency', 'Recency']])
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
sns.lineplot(x=k_list, y=sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE(Sum of Squared Errors)')
plt.xticks(k_list)
plt.grid(True)
plt.show()

# Step 7: KMeans 聚类
RFM_kmeans = KMeans(n_clusters=4, random_state=0).fit(RFM_data[['Monetary', 'Frequency', 'Recency']])
RFM_kmeans_normalized = KMeans(n_clusters=4, random_state=0).fit(RFM_data_normalized[['Monetary', 'Frequency', 'Recency']])

RFM_data['Cluster'] = RFM_kmeans.labels_
RFM_data_normalized['Cluster'] = RFM_kmeans_normalized.labels_

# Step 8: 层次聚类
linked = linkage(RFM_data[['Monetary', 'Frequency', 'Recency']], method='complete')
plt.figure(figsize=(10, 5))
dendrogram(linked, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

clusterCut = fcluster(linked, t=4, criterion='maxclust')
RFM_data['HC_Cluster'] = clusterCut

linked_normalized = linkage(RFM_data_normalized[['Monetary', 'Frequency', 'Recency']], method='complete')
plt.figure(figsize=(10, 5))
dendrogram(linked_normalized, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram normalized")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

clusterCut_normalized = fcluster(linked_normalized, t=4, criterion='maxclust')
RFM_data_normalized['HC_Cluster'] = clusterCut_normalized

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=8)
RFM_data_normalized['DBSCAN_Cluster'] = dbscan.fit_predict(RFM_data_normalized[['Monetary', 'Frequency', 'Recency']]).astype(str)

# Step 9: 可视化
fig = px.scatter_3d(RFM_data, x='Monetary', y='Frequency', z='Recency', color=RFM_data['Cluster'].astype(str))
fig.update_layout(title="3D plot Non-normalization")
fig.show()

fig = px.scatter_3d(RFM_data, x='Monetary', y='Frequency', z='Recency', color=RFM_data['HC_Cluster'].astype(str))
fig.update_layout(title="3D plot Hierarchical Clustering")
fig.show()

fig = px.scatter_3d(RFM_data_normalized, x='Monetary', y='Frequency', z='Recency', color=RFM_data_normalized['Cluster'].astype(str))
fig.update_layout(title="3D plot Normalization")
fig.show()

fig = px.scatter_3d(RFM_data_normalized, x='Monetary', y='Frequency', z='Recency', color=RFM_data_normalized['HC_Cluster'].astype(str))
fig.update_layout(title="3D plot Hierarchical Clustering Normalation")
fig.show()

fig_dbscan = px.scatter_3d(RFM_data_normalized,
    x='Monetary', y='Frequency', z='Recency', color='DBSCAN_Cluster',
    color_discrete_sequence=px.colors.qualitative.Plotly,
    title="3D plot DBSCAN Clustering (Normalized Data)")
fig_dbscan.update_layout(
    scene=dict(
        xaxis_title='Monetary (Normalized)',
        yaxis_title='Frequency (Normalized)',
        zaxis_title='Recency (Normalized)'
    ),
    legend_title_text='Cluster'
)
fig_dbscan.show()

def draw_3d_scatter(df, label, title, norm=False):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    clusters = df[label].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))
    for cluster, color in zip(clusters, colors):
        cluster_data = df[df[label] == cluster]
        ax.scatter(
            cluster_data['Monetary'], 
            cluster_data['Frequency'], 
            cluster_data['Recency'],
            c=[color], label=f'Cluster {cluster}',
            s=40, alpha=0.7, edgecolors='k', linewidths=0.3
        )
    suffix = " (Normalized)" if norm else ""
    ax.set_xlabel(f'Monetary{suffix}')
    ax.set_ylabel(f'Frequency{suffix}')
    ax.set_zlabel(f'Recency{suffix}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.show()

draw_3d_scatter(RFM_data, 'Cluster', 'KMeans Clustering (Original Data)')
draw_3d_scatter(RFM_data, 'HC_Cluster', 'Hierarchical Clustering (Original Data)')
draw_3d_scatter(RFM_data_normalized, 'Cluster', 'KMeans Clustering (Normalized Data)', norm=True)
draw_3d_scatter(RFM_data_normalized, 'HC_Cluster', 'Hierarchical Clustering (Normalized Data)', norm=True)
draw_3d_scatter(RFM_data_normalized, 'DBSCAN_Cluster', 'DBSCAN Clustering (Normalized Data)', norm=True)

# 饼图部分
num_clusters = RFM_data_normalized['Cluster'].nunique()
color_palette = sns.color_palette('pastel', num_clusters).as_hex()

size_stats = RFM_data_normalized.groupby('Cluster').agg(
    Size=('CustomerID', 'count')
).reset_index()

monetary_stats = RFM_data.copy()
monetary_stats['Cluster'] = RFM_data_normalized['Cluster']
monetary_stats = monetary_stats.groupby('Cluster').agg(
    Monetary=('Monetary', 'sum')
).reset_index()
monetary_stats['Monetary_pct'] = (monetary_stats['Monetary'] / monetary_stats['Monetary'].sum() * 100).round(1)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.pie(
    size_stats['Size'],
    labels=[f'Cluster {i}' for i in size_stats['Cluster']],
    autopct='%1.1f%%',
    startangle=90,
    colors=color_palette
)
plt.title("Cluster Size Distribution\n(Normalized KMeans Clusters)")

plt.subplot(122)
plt.pie(
    monetary_stats['Monetary'],
    labels=[f"{p}%" for p in monetary_stats['Monetary_pct']],
    startangle=90,
    colors=color_palette
)
plt.title("Cluster Revenue Distribution\n(Based on Original Monetary)")

plt.tight_layout()
plt.show()

RFM_data_normalized.to_csv("/mnt/e/Desktop/new/RFM_dataset.csv", index=False)