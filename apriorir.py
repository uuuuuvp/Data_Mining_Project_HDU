import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules  # 主要修改点：使用 Apriori

# 读取数据
retail_df = pd.read_excel(r"C:\Users\Zombie\Desktop\大三下\数据结构实验\课程大报告\清洗后数据.xlsx")

# 数据预处理（保持不变）
retail_df['Description'] = retail_df['Description'].str.strip()
retail_df.dropna(subset=['InvoiceNo'], inplace=True)
retail_df['InvoiceNo'] = retail_df['InvoiceNo'].astype(str)
retail_df = retail_df[~retail_df['InvoiceNo'].str.contains('C')]

# 二值化函数（返回布尔值）
def encode_units(x):
    return x > 0  # 返回 True/False 代替 0/1，提高性能

# 生成购物篮格式函数（保持不变）
def create_basket(country_filter):
    basket = (retail_df[retail_df['Country'] == country_filter]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    return basket

# 分析函数（主要修改点：使用 Apriori 算法）
def market_basket_analysis(country, support=0.05, lift=1.2, quantity_filter=None):
    print(f"\n{country} Market Basket Analysis Results (Apriori):")  # 修改输出提示
    if quantity_filter:
        data = retail_df[(retail_df['Country'] == country) & (retail_df['Quantity'] < quantity_filter)]
    else:
        data = retail_df[retail_df['Country'] == country]
    
    # 检查数据量
    if len(data) < 50:
        print(f"⚠️ 数据不足（仅{len(data)}条交易），可能无法生成有效规则")
        return
    
    basket = (data.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))

    basket_sets = basket.map(encode_units)  # 使用 map 替代 applymap
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE', axis=1, inplace=True)

    # 使用 Apriori 算法替代 FP-Growth
    frequent_itemsets = apriori(basket_sets, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift)
    
    # 结果输出
    if not rules.empty:
        print(f"发现 {rules.shape[0]} 条关联规则")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
              .sort_values('lift', ascending=False).head(10))
        
        # 可视化
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=rules, x='support', y='confidence', size='lift', hue='lift',
                        palette='viridis', sizes=(50, 500), alpha=0.7, edgecolor='k')
        plt.title(f"{country} Market Basket Rules (Apriori)")  # 修改标题
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.legend(title='Lift', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 未发现关联规则，建议尝试：\n"
              f"1. 降低 min_support（当前={support}）\n"
              f"2. 降低 min_lift（当前={lift}）")

# 调用分析（参数可根据需要调整）
market_basket_analysis("France", support=0.03)  # 法国市场，支持度 3%
market_basket_analysis("Germany", support=0.04, lift=1.0)  # 德国市场，支持度 4%，提升度 1.0
market_basket_analysis("United Kingdom", support=0.02)  # 英国市场，支持度 2%