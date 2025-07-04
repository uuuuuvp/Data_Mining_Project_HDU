import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
classification_data = pd.read_csv("/mnt/e/Desktop/new/RFM_dataset.csv")
if 'Unnamed: 0' in classification_data.columns:
    classification_data.drop(columns=['Unnamed: 0'], inplace=True)

classification_data['Cluster'] = classification_data['Cluster'].astype(str)
X = classification_data.drop(columns=['Cluster', 'CustomerID', 'HC_Cluster', 'DBSCAN_Cluster'], errors='ignore')
y = classification_data['Cluster']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.23, random_state=42, stratify=y
)

# 参数范围
n_estimators_range = list(range(2, 11))
max_depth_range = list(range(2, 7))

# 准备坐标和 F1 值
f1_matrix = np.zeros((len(max_depth_range), len(n_estimators_range)))

for i, max_depth in enumerate(max_depth_range):
    for j, n in enumerate(n_estimators_range):
        clf = RandomForestClassifier(
            n_estimators=n,
            max_depth=max_depth,
            min_samples_leaf=1,
            criterion='entropy',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_matrix[i, j] = f1

# 准备网格数据用于3D图
X_grid, Y_grid = np.meshgrid(n_estimators_range, max_depth_range)
Z = f1_matrix

# 绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D曲面
surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', edgecolor='k', linewidth=0.5)

ax.set_xlabel('n_estimators (num of trees)')
ax.set_ylabel('max_depth')
ax.set_zlabel('F1 Score')
ax.set_title('RandomForest :F1-Score(test set) n_estimators and max_depth')

# 添加 colorbar 便于读值
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()

