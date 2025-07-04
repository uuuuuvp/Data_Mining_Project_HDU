import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

# 1. 数据加载与预处理
def load_and_preprocess(filepath):
    """加载并预处理零售数据"""
    retail_df = pd.read_excel(filepath)
    
    # 基础清洗
    retail_df['Description'] = retail_df['Description'].str.strip()
    retail_df.dropna(subset=['InvoiceNo'], inplace=True)
    retail_df['InvoiceNo'] = retail_df['InvoiceNo'].astype(str)
    
    # 移除退货订单（以"C"开头的发票号）
    retail_df = retail_df[~retail_df['InvoiceNo'].str.contains('C')]
    
    return retail_df

# 2. 购物篮数据准备
def prepare_country_basket(retail_df, country):
    """生成指定国家的购物篮矩阵"""
    country_df = retail_df[retail_df['Country'] == country]
    
    basket = (country_df.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    
    # 布尔转换函数
    def encode_units(x):
        return x > 0
    
    basket_bool = basket.applymap(encode_units)
    
    # 移除非商品列（如邮费）
    non_product_cols = ['POSTAGE', 'DOTCOM POSTAGE', 'CRUK Commission']
    for col in non_product_cols:
        if col in basket_bool.columns:
            basket_bool.drop(col, axis=1, inplace=True)
    
    return basket_bool

# 3. 关联规则分析与可视化
def analyze_and_visualize(basket_data, country, min_support=0.02, min_lift=1):
    """执行FP-Growth分析并生成可视化"""
    # 使用FP-Growth算法
    frequent_itemsets = fpgrowth(basket_data, 
                               min_support=min_support, 
                               use_colnames=True)
    
    rules = association_rules(frequent_itemsets, 
                            metric="lift", 
                            min_threshold=min_lift)
    
    if rules.empty:
        print(f"⚠️ 未发现满足条件的关联规则（support>={min_support}, lift>={min_lift}）")
        return None
    
    # 按lift值排序并选择前20条规则
    top_rules = rules.sort_values('lift', ascending=False).head(20)
    
    # 3.1 散点图可视化
    plt.figure(figsize=(12, 6))
    scatter = sns.scatterplot(
        data=top_rules,
        x='support',
        y='confidence',
        size='lift',
        hue='lift',
        palette='viridis',
        sizes=(50, 300),
        alpha=0.7,
        edgecolor='k'
    )
    plt.title(f"{country} Market Basket Rules\n(Support ≥ {min_support}, Lift ≥ {min_lift})")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.legend(title='Lift', bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 3.2 热力图可视化
    # 获取所有涉及的商品
    all_products = set()
    for itemset in top_rules['antecedents']:
        all_products.update(itemset)
    for itemset in top_rules['consequents']:
        all_products.update(itemset)
    all_products = sorted(all_products)
    
    # 创建关联矩阵
    heatmap_matrix = pd.DataFrame(
        np.zeros((len(all_products), len(all_products))),
        index=all_products,
        columns=all_products
    )
    
    # 填充矩阵（使用lift值）
    for _, row in top_rules.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                heatmap_matrix.loc[ant, con] = row['lift']
    
    # 绘制热力图
    plt.figure(figsize=(12, 9))
    heatmap = sns.heatmap(
        heatmap_matrix,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={'label': 'Lift Value'},
        annot_kws={"size": 8}
    )
    plt.title(
        f"{country} Product Association Heatmap\n(Support ≥ {min_support}, Lift ≥ {min_lift})",
        pad=20,
        fontsize=14
    )
    plt.xlabel("Consequents (Recommended Products)", fontsize=12)
    plt.ylabel("Antecedents (Purchased Products)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return top_rules

# 主执行函数
def main():
    # 文件路径（需修改为实际路径）
    file_path = r"C:\Users\Zombie\Desktop\大三下\数据结构实验\课程大报告\清洗后数据.xlsx"
    
    # 加载数据
    retail_data = load_and_preprocess(file_path)
    
    # 选择国家（这里以英国为例）
    target_country = "United Kingdom"
    
    # 准备购物篮数据
    uk_basket = prepare_country_basket(retail_data, target_country)
    
    # 分析参数设置
    min_support = 0.02  # 最小支持度
    min_lift = 1.2        # 最小提升度
    
    # 执行分析并可视化
    results = analyze_and_visualize(
        uk_basket,
        country=target_country,
        min_support=min_support,
        min_lift=min_lift
    )
    
    # 输出结果摘要
    if results is not None:
        print("\nTop Association Rules:")
        print(results[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
              .sort_values('lift', ascending=False).head(10))

if __name__ == "__main__":
    main()