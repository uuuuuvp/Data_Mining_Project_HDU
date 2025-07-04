import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
import os

# 禁用 Qt 插件调试信息
os.environ['QT_DEBUG_PLUGINS'] = '0'

# Step 1: 读取数据
classification_data = pd.read_csv("/mnt/e/Desktop/new/RFM_dataset.csv")
if 'Unnamed: 0' in classification_data.columns:
    classification_data = classification_data.drop(columns=['Unnamed: 0'])

# 目标变量转字符串类型（多分类）
classification_data['Cluster'] = classification_data['Cluster'].astype(str)

# Step 2: 特征与标签
X = classification_data.drop(columns=['Cluster', 'CustomerID', 'HC_Cluster', 'DBSCAN_Cluster'], errors='ignore')
y = classification_data['Cluster']

# Step 3: 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.23, random_state=42, stratify=y
)

# Step 4: 构建随机森林
forest = RandomForestClassifier(
    n_estimators=3,
    max_depth=4,
    min_samples_leaf=1,
    criterion='entropy',
    random_state=0,
    n_jobs=-1
)
forest.fit(X_train, y_train)

# Step 5: 预测与评估
train_pred = forest.predict(X_train)
test_pred = forest.predict(X_test)

print("\n=== 训练集分类报告 ===")
print(classification_report(y_train, train_pred, digits=4))
print(f"训练集准确率: {accuracy_score(y_train, train_pred):.4f}")
print(f"训练集加权F1: {f1_score(y_train, train_pred, average='weighted'):.4f}")

print("\n=== 测试集分类报告 ===")
print(classification_report(y_test, test_pred, digits=4))
print(f"测试集准确率: {accuracy_score(y_test, test_pred):.4f}")
print(f"测试集加权F1: {f1_score(y_test, test_pred, average='weighted'):.4f}")

# Step 6: 打印前3棵树的结构（文本）
for i, tree in enumerate(forest.estimators_[:4]):
    print(f"\n=== 随机森林中的决策树 {i+1} 结构（文本表示） ===")
    tree_rules = export_text(tree, feature_names=list(X.columns))
    print(tree_rules)

# Step 7: 可视化前3棵决策树（图形）
for i, tree in enumerate(forest.estimators_[:3]):
    plt.figure(figsize=(20, 12))  # 放大图像
    plot_tree(
        tree,
        feature_names=X.columns,
        class_names=forest.classes_,
        filled=True,
        rounded=True,
        max_depth=4,  # 可根据需要缩小深度
        fontsize=8  # 减小字体防止重叠
    )
    plt.title(f"Decision Tree {i+1} from Random Forest", fontsize=14)
    plt.tight_layout()
    plt.show()

