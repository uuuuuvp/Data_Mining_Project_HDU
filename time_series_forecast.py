# -*- coding: utf-8 -*-
"""
基于每日销售额的时间序列预测与评估（优化版2）
-------------------------------------------------
优化内容：
1. 分别绘制ARIMA和Prophet的拟合对比图
2. 改进评估结果展示方式
3. 优化未来预测的可视化
"""

import sys
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
plt.rcParams["axes.unicode_minus"] = False

# ===========================================================
# 1. 数据加载与预处理
# ===========================================================

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """加载 Excel 数据并返回每日销售额 DataFrame(ds, y)"""
    df = pd.read_excel(filepath, parse_dates=["InvoiceDate"])

    # 数据清洗
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    
    # 计算销售额并处理异常值
    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    q_low, q_high = df["Amount"].quantile([0.01, 0.99])
    df = df[(df["Amount"] >= q_low) & (df["Amount"] <= q_high)]

    # 按天聚合并平滑处理
    daily_sales = (
        df.groupby(pd.Grouper(key="InvoiceDate", freq="D"))["Amount"]
          .sum()
          .replace(0, np.nan)
          .rolling(7, min_periods=1).mean()
          .reset_index()
          .rename(columns={"InvoiceDate": "ds", "Amount": "y"})
          .dropna()
    )
    return daily_sales

# ===========================================================
# 2. 可视化
# ===========================================================

def plot_sales_trend(data: pd.DataFrame):
    """绘制销售趋势和分布图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 销售趋势
    ax1.plot(data["ds"], data["y"], label="每日销售额", linewidth=1, color="tab:blue")
    ax1.set_title("历史销售趋势", fontsize=16, pad=20)
    ax1.set_xlabel("日期", labelpad=10)
    ax1.set_ylabel("销售额", labelpad=10)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 销售分布
    ax2.hist(data["y"], bins=50, color="tab:orange", alpha=0.7)
    ax2.set_title("销售额分布", fontsize=16, pad=20)
    ax2.set_xlabel("销售额区间", labelpad=10)
    ax2.set_ylabel("出现频次", labelpad=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ===========================================================
# 3. 预测模型
# ===========================================================

def arima_forecast(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """使用AutoARIMA进行预测"""
    sf_train = train.assign(unique_id=1)
    sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=-1)
    forecast_df = sf.forecast(df=sf_train, h=horizon, level=[90])
    for col in ["AutoARIMA", "AutoARIMA-lo-90", "AutoARIMA-hi-90"]:
        forecast_df[col] = forecast_df[col].clip(lower=0)
    return forecast_df.rename(columns={
        "AutoARIMA": "yhat", 
        "AutoARIMA-lo-90": "yhat_lower",
        "AutoARIMA-hi-90": "yhat_upper"
    })

def prophet_forecast(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """使用Prophet进行预测"""
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    try:
        model.add_country_holidays(country_name="CN")
    except Exception:
        pass

    model.fit(train)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)

# ===========================================================
# 4. 评估与可视化
# ===========================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算对称平均绝对百分比误差"""
    return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def plot_model_fit(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    pred: pd.DataFrame,
    model_name: str,
    color: str
):
    """绘制单个模型的拟合情况"""
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # 绘制历史数据
    ax.plot(train["ds"], train["y"], label="训练数据", color="gray", alpha=0.7)
    
    # 绘制实际测试值
    ax.plot(test["ds"], test["y"], label="实际值", color="black", linewidth=2)
    
    # 绘制预测值
    ax.plot(pred["ds"], pred["yhat"], 
            label=f"{model_name}预测", linestyle="--", color=color)
    ax.fill_between(
        pred["ds"], 
        pred["yhat_lower"], 
        pred["yhat_upper"], 
        color=color, 
        alpha=0.1,
        label="90%置信区间"
    )
    
    # 设置图表样式
    ax.set_title(f"{model_name}模型拟合情况", fontsize=16, pad=20)
    ax.set_xlabel("日期", labelpad=10)
    ax.set_ylabel("销售额", labelpad=10)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_future_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    model_name: str,
    color: str
):
    """绘制未来预测结果"""
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # 绘制历史数据
    ax.plot(history["ds"], history["y"], label="历史数据", color="gray")
    
    # 绘制预测值
    ax.plot(forecast["ds"], forecast["yhat"], 
            label=f"{model_name}预测", color=color)
    ax.fill_between(
        forecast["ds"], 
        forecast["yhat_lower"], 
        forecast["yhat_upper"], 
        color=color, 
        alpha=0.2,
        label="90%置信区间"
    )
    
    # 标记预测开始点
    last_date = history["ds"].max()
    ax.axvline(x=last_date, color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"预测开始\n{last_date.strftime('%Y-%m-%d')}", 
                xy=(last_date, history["y"].max()*0.8),
                xytext=(10, 0), textcoords="offset points",
                bbox=dict(boxstyle="round", alpha=0.1))
    
    # 设置图表样式
    ax.set_title(f"{model_name}未来30天销售额预测", fontsize=16, pad=20)
    ax.set_xlabel("日期", labelpad=10)
    ax.set_ylabel("销售额", labelpad=10)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_models(
    data: pd.DataFrame, 
    test_days: int = 30
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame, pd.DataFrame]:
    """评估模型性能并返回指标和预测结果"""
    train, test = data.iloc[:-test_days].copy(), data.iloc[-test_days:].copy()
    y_true = test["y"].values
    
    # ARIMA预测
    arima_pred = arima_forecast(train, horizon=test_days)
    arima_pred = pd.merge(test[["ds"]], arima_pred, on="ds", how="left")
    
    # Prophet预测
    prophet_pred = prophet_forecast(train, horizon=test_days)
    prophet_pred = pd.merge(test[["ds"]], prophet_pred, on="ds", how="left")
    
    # 计算指标
    metrics = {
        "ARIMA": {
            "MAE": mean_absolute_error(y_true, arima_pred["yhat"]),
            "RMSE": np.sqrt(mean_squared_error(y_true, arima_pred["yhat"])),
            "sMAPE": smape(y_true, arima_pred["yhat"]),
        },
        "Prophet": {
            "MAE": mean_absolute_error(y_true, prophet_pred["yhat"]),
            "RMSE": np.sqrt(mean_squared_error(y_true, prophet_pred["yhat"])),
            "sMAPE": smape(y_true, prophet_pred["yhat"]),
        },
    }
    
    return metrics, arima_pred, prophet_pred

def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """打印评估指标"""
    print("\n=== 模型评估结果 ===")
    print(f"{'指标':<10}{'ARIMA':<15}{'Prophet':<15}")
    print("-" * 40)
    for metric in ["MAE", "RMSE", "sMAPE"]:
        print(f"{metric:<10}{metrics['ARIMA'][metric]:<15.2f}{metrics['Prophet'][metric]:<15.2f}")

# ===========================================================
# 5. 主流程
# ===========================================================

def main(file_path: str):
    # 1. 加载数据
    print("正在加载数据 ...")
    daily_sales = load_and_preprocess(file_path)
    print(
        f"\n数据时间范围: {daily_sales['ds'].min().date()} 至 "
        f"{daily_sales['ds'].max().date()}\n"
        f"总天数: {len(daily_sales)}\n"
        f"日均销售额: {daily_sales['y'].mean():.2f} ± "
        f"{daily_sales['y'].std():.2f}"
    )

    # 2. 可视化历史数据
    plot_sales_trend(daily_sales)

    # 3. 模型评估与比较
    metrics, arima_pred, prophet_pred = evaluate_models(daily_sales, test_days=30)
    
    # 打印评估结果
    print_metrics(metrics)
    
    # 绘制拟合情况图
    train_data = daily_sales.iloc[:-30]
    test_data = daily_sales.iloc[-30:]
    
    plot_model_fit(
        train=train_data,
        test=test_data,
        pred=arima_pred,
        model_name="ARIMA",
        color="red"
    )
    
    plot_model_fit(
        train=train_data,
        test=test_data,
        pred=prophet_pred,
        model_name="Prophet",
        color="blue"
    )

    # 4. 未来预测
    print("\n=== 未来30天预测 ===")
    
    # ARIMA未来预测
    arima_future = arima_forecast(daily_sales, horizon=30)
    print("\nARIMA预测结果:")
    print(arima_future.head())
    plot_future_forecast(
        history=daily_sales,
        forecast=arima_future,
        model_name="ARIMA",
        color="red"
    )
    
    # Prophet未来预测
    prophet_future = prophet_forecast(daily_sales, horizon=30)
    print("\nProphet预测结果:")
    print(prophet_future.head())
    plot_future_forecast(
        history=daily_sales,
        forecast=prophet_future,
        model_name="Prophet",
        color="blue"
    )

if __name__ == "__main__":
    # 默认文件路径
    DEFAULT_FILE = Path(r"清洗后数据.xlsx")

    # 处理命令行参数
    if len(sys.argv) == 2:
        excel_path = Path(sys.argv[1])
        print("检测到命令行参数，使用指定文件：", excel_path)
    else:
        excel_path = DEFAULT_FILE
        print("未检测到命令行参数，使用默认文件：", excel_path)

    # 文件存在性检查
    if not excel_path.exists():
        sys.exit(f"❌ 文件不存在：{excel_path}")

    main(str(excel_path))