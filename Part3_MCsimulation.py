# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:13:22 2025

@author: user
"""

"""
學號：B3871571
姓名：白若善
Python程式設計  自主學習 -- 專題製作
第三部份：以Monte Carlo Simulation預測卜蜂未來一年的股價趨勢
"""

#========================================================================
# 匯入所需模組及套件
#========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#========================================================================
# 先讀出資料，並再檢視一次
#========================================================================
cpfood = pd.read_csv(r"./data/cpfood.csv")

print('數據基本信息：')
cpfood.info()
rows, columns = cpfood.shape

#========================================================================
# 設定模擬所需參數
#========================================================================
adj_close = cpfood["Adj Close"]
returns = adj_close.pct_change().dropna()
mu = returns.mean()
sigma = returns.std()
num_simulations = 10000  # 模擬次數設為1萬次
num_days = 252
last_price = adj_close.iloc[-1]

#========================================================================
# 進行10,000次Monte Carlo模擬
# 將模擬結果存成Numpy的陣列型態
#========================================================================
simulation_results = np.zeros((num_simulations, num_days))

for i in range(num_simulations):
    price_series = [last_price]
    for _ in range(num_days):
        random_return = np.random.normal(mu, sigma)
        new_price = price_series[-1] * (1 + random_return)
        price_series.append(new_price)
    simulation_results[i, :] = price_series[1:]

#========================================================================
# 將模擬結果視覺化輸出
#========================================================================
plt.figure(figsize=(15, 7))
for i in range(num_simulations):
    plt.plot(simulation_results[i, :], alpha=0.1)

plt.title("Monte Carlo Simulation", fontsize=20)
plt.xlabel("Days")
plt.xticks(rotation=45)
plt.ylabel("Stock Prices")
plt.show()

#========================================================================
# 將模擬結果以DataFrame的格式輸出
#========================================================================
average_prices = np.mean(simulation_results, axis=0)
max_prices = np.max(simulation_results, axis=0)
min_prices = np.min(simulation_results, axis=0)

result_df = pd.DataFrame({
    "交易日": range(1, num_days + 1),
    "日平均價": average_prices,
    "最高價": max_prices,
    "最低價": min_prices,
    "日報酬率平均值": mu,
    "日報酬率標準差": sigma    
    })

print(result_df)

#========================================================================
# 使用直方圖觀察模擬結果的分配情形 -- 常態分佈 vs. 非常態分佈
# 並以np.argmax()的方法，取出最可能出現的股價區間
#========================================================================
flattened_simulations = simulation_results.flatten()
hist, bin_edges = np.histogram(simulation_results, bins=50)
max_bin_index = np.argmax(hist)
most_likely_range = [(bin_edges[max_bin_index], bin_edges[max_bin_index + 1])]
print(f"最可能區間為: {most_likely_range}")

plt.figure(figsize=(15, 7))
plt.hist(flattened_simulations, bins=50, edgecolor='black')
plt.title("Histogram of Continuous Simulation Results", fontsize=20)
plt.xlabel("Stock Prices")
plt.ylabel("Frequency")
plt.show()

#========================================================================
# 小結：
# 1. 直方圖顯示，10,000次的模擬結果，卜蜂未來股價並非呈現正常的常態分佈
# 2. 模擬結果顯示，股價呈現左偏分佈
# 3. 而且出現很明顯的右邊長尾效應-->mean = 136時，右邊最大值約為300
# 4. 一年後最可能出現的股價區間，為每股108元至117元。
# 5. Monte Carlo Simulation出現和CAPM近似的結論：
# - 卜蜂為一檔防禦型個股
# - 以最可能出現的股價區間推論，未來一年卜蜂幾乎沒有下跌的空間
# - 但是有5%至24%的上漲幅度，在極端值出現時，卜蜂可能漲到每股NT$300
#========================================================================







