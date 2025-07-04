# 🛒 Online Retail Data Mining Project（本README即为软件使用说明）

本项目基于真实的电商交易数据集（Online Retail Dataset），应用数据挖掘与机器学习技术，完成了客户价值挖掘、购物箱分析、分类建模以及销售预测等多个实用分析任务。

---

## 🧠 项目目的

本项目旨在通过挖掘和分析电商交易数据，实现以下目标：

- 构建 RFM 模型并进行聚类，识别客户价值分层；
- 利用随机森林算法进行客户分类与建模；
- 通过 Apriori 与 FP-Growth 算法分析商品间的关联规则；
- 应用时间序列模型（ARIMA 与 Prophet）预测未来销售趋势；
- 使用多种可视化方式，增强数据测观与模型解释性。

---

## 📁 项目结构

```
├— apriorir.py                   # Apriori 关联规则分析
├— FP-growth.py                  # FP-Growth 关联规则分析
├— RFM_cluster.py                # RFM 聚类分析
├— RFM_classification.py         # 客户分类（随机梯）
├— seleect_randomforest_parameter.py # 随机梯参数调优
├— time_series_forecast.py       # 销售额时间序列预测（ARIMA + Prophet）
├— online_retail.csv             # 源数据（CSV 格式）
├— Online Retail.xlsx            # 源数据（Excel 格式）
├— requirements.txt              # 依赖包列表
├— LICENSE                       # 项目授权协议
└— README.md                     # 项目说明文件（当前）
```

---

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

推荐使用 Python 3.7，并确保安装如下关键依赖：

- pandas, numpy, seaborn, matplotlib
- scikit-learn, mlxtend
- prophet, statsforecast

---

## 🧰 各模块说明

### 1️⃣ RFM 聚类分析（`RFM_cluster.py`）
- 根据 Recency、Frequency、Monetary 三个维度构建 RFM 模型。
- 采用 KMeans、层次聚类、DBSCAN 三种方法对客户进行聚类。
- 可视化聚类结果，便于客户分层管理。

### 2️⃣ 随机梯参数选择（`seleect_randomforest_parameter.py`）
- 对随机梯的关键参数（如 `n_estimators`, `max_depth`）进行调优。
- 使用网格搜索方式评估模型性能，辅助选择最优组合。

### 3️⃣ 客户分类模型（`RFM_classification.py`）
- 使用随机梯算法对聚类结果进行多分类建模。
- 评估分类模型性能（准确率、F1 值等）。
- 输出每梯子树结构并可视化部分决策树。

### 4️⃣ 商品组合推荐（Apriori / FP-Growth 关联规则挖掘）
借助Apriori与FP-Growth算法，挖掘频繁商品组合及强关联规则。例如：
> 买了 “蜡烛” 的顾客，可能也会买 “香薰” 或 “玻璃瓶”  
这种分析可用于：
- 购物车推荐
- 捆绑促销策略优化

### 5️⃣ 销售额时间序列预测（`time_series_forecast.py`）
- 对历史销售数据进行平滑处理，构建每日销售时间序列。
- 使用 AutoARIMA 与 Prophet 两种模型进行预测与比较：
  - ARIMA：基于自回归滑动平均原理自动拟合。
  - Prophet：由 Meta 开发，支持季节性与节假日建模。
- 输出评估指标（MAE, RMSE, sMAPE）及未来30天预测结果。
- 支持训练拟合图、未条预测图可视化。

---

## 💠 使用方式（示例）

```bash
# 客户聚类
python RFM_cluster.py

# 随机梯建模
python RFM_classification.py
# 依赖聚类生成的文件来分类

# 参数选择
python seleect_randomforest_parameter.py

# FP-Growth 分析
python FP-growth.py

# Apriori 分析
python apriorir.py

# 销售额预测
python time_series_forecast.py "Online Retail.xlsx"
```

---

### ⚙️ 项目亮点与技术融合

- ✅ **多模型协同**：聚类、分类、预测等功能分别引入多种算法互补组合，确保分析稳定性与准确性
- ✅ **自动特征工程**：自动筛选高价值变量，避免冗余信息干扰，提高运行效率
- ✅ **可视化支持强**：输出丰富图表（热力图、散点图、趋势图、决策树图），便于业务理解与汇报展示
- ✅ **一键复现**：所有功能模块已打包，使用说明清晰，便于快速上手运行和部署

---