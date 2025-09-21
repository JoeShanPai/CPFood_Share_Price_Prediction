# -*- coding: utf-8 -*-
"""
Created on Sat May 31 22:09:27 2025

@author: user
"""

"""
學號：B3871571
姓名：白若善
Python程式設計  自主學習 -- 專題製作
第二部份：以CAPM預測卜蜂的投資報酬率及未來一年股價
- 基本假設(H0)>> ETF 0050可作為台灣股市(含上市、上櫃、興櫃)的 proxy。
"""

#========================================================================
# 匯入所需模組及套件
#========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

#=========================================================================
# 檢視資料是否為所需的內容及格式，將Date作為索引欄
# 假設(H0假設)：ETF 0050是台灣證券市場的proxy(完整的替代標的)；
# 此處證券市場包含集中交易市場、櫃檯買賣市場、興櫃市場、創業板、及場外鉅額交易
#=========================================================================
etf0050 = pd.read_csv(r"./data/etf0050.csv")
etf0050.set_index("Date", inplace=True)
print(etf0050.head())
print(etf0050.tail())
print(etf0050.info())

print("="*60)

cpfood = pd.read_csv(r"./data/cpfood.csv")
cpfood.set_index("Date", inplace=True)
print(cpfood.head())
print(cpfood.tail())
print(cpfood.info())

#=========================================================================
# 本段程式分兩大部份，前段處理ETF0050，後段處理卜蜂(CP Food)
# 以調整除權息後收盤價"Adj Close"，計算簡單歷史投資報酬率
# 將計算結果視覺化輸出，創建兩種圖：
# 1. 時間序列圖
# 2. Daily Simple Return的分佈直方圖
#=========================================================================
etf0050["Simple Return"]=etf0050["Adj Close"]/etf0050["Adj Close"].shift(1)-1 
#--->此為單日歷史報酬率，亦即單日漲跌幅
print(etf0050["Simple Return"])
plt.figure(figsize=(10,5))
etf0050["Simple Return"].plot()
plt.title("ETF 0050 Daily Simple Return", fontsize=18)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
etf0050["Simple Return"].plot(ax=ax1, color='blue', alpha=0.7)
ax1.set_title("ETF 0050 Daily Simple Return", fontsize=16)
ax1.set_ylabel("Return", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

n, bins, patches = ax2.hist(etf0050["Simple Return"].dropna(), 
                           bins=50, 
                           density=True, 
                           alpha=0.7, 
                           color='skyblue',
                           edgecolor='black')

# 繪製一個常態分配曲線，對照參考
mu = etf0050["Simple Return"].mean()
sigma = etf0050["Simple Return"].std()
y = norm.pdf(bins, mu, sigma)
ax2.plot(bins, y, 'r--', linewidth=2)
ax2.set_title("Distribution of Daily Returns", fontsize=16)
ax2.set_xlabel("Return", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)


cpfood["Simple Return"]=cpfood["Adj Close"]/cpfood["Adj Close"].shift(1)-1 
print(cpfood["Simple Return"])
plt.figure(figsize=(10,5))
cpfood["Simple Return"].plot()
plt.title("CP Food Daily Simple Return", fontsize=18)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

cpfood["Simple Return"].plot(ax=ax1, color='blue', alpha=0.7)
ax1.set_title("CP Food Daily Simple Return", fontsize=16)
ax1.set_ylabel("Return", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

n, bins, patches = ax2.hist(cpfood["Simple Return"].dropna(), 
                           bins=50, 
                           density=True, 
                           alpha=0.7, 
                           color='skyblue',
                           edgecolor='black')

mu = cpfood["Simple Return"].mean()
sigma = cpfood["Simple Return"].std()
y = norm.pdf(bins, mu, sigma)
ax2.plot(bins, y, 'r--', linewidth=2)
ax2.set_title("Distribution of Daily Returns", fontsize=16)
ax2.set_xlabel("Return", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
#===================================================================================
# 小結一：
# 從matplotlib功能繪製的圖型觀察：
# 從2008/1/2至2025/5/28的期間內
# ETF 0050及卜蜂的單日漲跌幅，均呈現為常態分佈(Normal distribution)
#====================================================================================

#=========================================================================
# 將單日漲跌幅轉換成年化報酬率
# 計算年化報酬率的平均值(mean)、變異數(variance)、及標準差(standard deviation)
# 每年有252個交易日，為台灣證券業約定成俗的假設
# 將Normalized後的兩檔標的股價歷史走勢報酬率，以line chart輸出
#==========================================================================
etf0050_annual_return = (1+etf0050["Simple Return"]).prod()**(252/len(etf0050))-1
etf0050_annual_return_mean = etf0050["Simple Return"].mean()*252
etf0050_annual_return_var = etf0050["Simple Return"].var()*252
etf0050_annual_return_std = etf0050_annual_return_var**0.5
print(f"ETF 0050的歷史年化報酬率 = {etf0050_annual_return*100:.4f}%")
print(f"ETF 0050歷史報酬率的平均數 = {etf0050_annual_return_mean*100:.4f}%")
print(f"ETF 0050歷史報酬率的變異數 = {etf0050_annual_return_var:.4f}")
print(f"ETF 0050歷史報酬率的標準差 = {etf0050_annual_return_std*100:.4f}%")

print("="*60)

cpfood_annual_return = (1+cpfood["Simple Return"]).prod()**(252/len(cpfood))-1 
cpfood_annual_return_mean = cpfood["Simple Return"].mean()*252
cpfood_annual_return_var = cpfood["Simple Return"].var()*252
cpfood_annual_return_std = cpfood_annual_return_var**0.5 
print(f"卜蜂歷史年化報酬率 = {cpfood_annual_return*100:.4f}%")
print(f"卜蜂歷史報酬率的平均數 = {cpfood_annual_return_mean*100:.4f}%")
print(f"卜蜂歷史報酬率的變異數 = {cpfood_annual_return_var:.4f}")
print(f"卜蜂歷史報酬率的標準差 = {cpfood_annual_return_std*100:.4f}%")

# --> 以下將兩個表格的Adj Close取出，合併成一個新的DataFrame
merged = pd.concat([etf0050["Adj Close"],cpfood["Adj Close"]], axis=1)
merged.columns = ["etf0050_Adj Close", "cpfood_Adj Close"]
csv_path = r"./data/portfolio.csv"               #合併後存入portfolio.csv
merged.to_csv(csv_path)
portfolio = pd.read_csv(r"./data/portfolio.csv") 
print(portfolio.info())                          #檢視資料
portfolio = portfolio[:-3]                       #再次清洗資料
portfolio.to_csv(csv_path, index=False)          #覆蓋掉舊的檔案
print(portfolio.info())                          #再檢視一次資料
# --> 處理完成，下面用視覺化呈現 -->
numeric_col = portfolio.select_dtypes(include="number").columns
normalized_portfolio = portfolio[numeric_col]/portfolio[numeric_col].iloc[0]*100
sharePrice_chart = normalized_portfolio.plot(figsize=(12,6))
plt.title("Normalized Share Price Performance - CPFood vs. ETF0050", fontsize=16)
plt.show()

#===================================================================================
# 小結二：
# 從年化報酬率、標準差、及視覺化圖觀察，
# 持有卜蜂單一個股的報酬，比持有被動式操作的指數型ETF（台灣50）要高
# 被動式、指數型、分散投資的基金管理方式，真的比主動式選股的績效好嗎？
#================================================================================

#==================================================================================
# 將卜蜂加入 ETF 0050的成份股中
# 基本假設：Fully-invested，亦即不持有現金，所有資金必須買進卜蜂及ETF0050
# 以亂數決定，分別持有卜蜂及ETF 0050的權重
# 重新觀察這個新的投資組合的表現
#===================================================================================
assets = ["etf0050", "cpfood"]
num_assets = len(assets)
#--> 兩個權重之和，必須是100%，亦即必須完全符合fully-invested的假設
weights = np.random.random(num_assets)
weights /= np.sum(weights)
print(round(weights[0]+weights[1],1))
# --> 個別持有權重處理完成
# --> 利用numpy功能，取自然對數值，計算預期報酬率，亦即報酬率的期望值
p_returns = np.log(portfolio[numeric_col]/portfolio[numeric_col].shift(1))
p_annual_returns = (1+p_returns).prod()**(252/len(portfolio))-1
print(p_annual_returns*100,"%")
p_mean = p_returns.mean()*252 
expected_return = np.sum(weights*p_mean) #權值為隨機產生，故每次生成的期望值必定不同
print(f"將卜蜂加入0050後，產生新的年化報酬率期望值 = {expected_return*100:.4f}%")
# --> 處理完成
# --> 計算共變異數(covariance)、預期變異數、預期標準差
p_cov = p_returns.cov()*252 
expected_var = np.dot(weights.T, np.dot(p_cov, weights))
print(f"將卜蜂加入0050後，產生新的變異數期望值 = {expected_var:.4f}")
expected_std = np.sqrt(np.dot(weights.T, np.dot(p_cov, weights)))
print(f"將卜蜂加入0050後，產生新的標準差期望值 = {expected_std*100:.4f}%")

expected_return = []
expected_std = []

# 將上列計算過程循環執行1,000次，模擬1,000種風險與報酬的組合
for x in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    exp_ret = np.sum(weights*p_mean)
    expected_return.append(exp_ret)
    exp_std = np.sqrt(np.dot(weights.T, np.dot(p_cov, weights)))
    expected_std.append(exp_std)
 
expected_return = np.array(expected_return)
expected_std = np.array(expected_std)

portfolio_statistic = pd.DataFrame({"Expected Return":expected_return, "Volatility":expected_std})
plt.figure(figsize=(12,6))
plt.scatter(expected_std, expected_return, marker="o", s=10, alpha=0.3)
plt.xlabel("Volatility") 
plt.xlim(0.18,0.30)
plt.ylabel("Expected Return")
plt.ylim(0,0.2)
plt.title("Markowitz Efficient Frontier", fontsize=20)
plt.show()

#===================================================================================
# 小結三：
# 在模擬形成的樣本數足夠的情形下(本例為1,000個樣本)：
# 任兩個標的形成的新投資組合，均可形成新的Markowitz Efficient Frontier
# 1. 否定H0假設
# 2. 持有這個新的組合，投資學上的解釋：比持有ETF 0050更好。
# 3. ETF 0050這項商品，是否仍有很多方面，需要改進？
#================================================================================

#==================================================================================
# 計算卜蜂的Beta值
# 基本假設：risk-free rate = 0.045, risk-premium = 0.10
# CAPM公式：Expected return = risk-free rate + Beta*risk premium
# 用一年期預期報酬率，計算一年後卜蜂股價
#===================================================================================
cov_cpfood_etf0050 = p_cov.iloc[0,1]
print(cov_cpfood_etf0050)
beta_cpfood = cov_cpfood_etf0050/etf0050_annual_return_var
print(beta_cpfood)

expected_return = 0.045 + beta_cpfood*0.10
print(f"預期未來一年持有卜蜂的報酬率 = {expected_return*100:.4f}%")
price_2025_05_29 = 111.00 
price_one_year_future = price_2025_05_29*(1+expected_return)
print(f"預測一年後卜蜂股價為 = NT${price_one_year_future:.2f}")

#===================================================================================
# 小結四：
# 卜蜂的Beta值遠低於+1，證明卜蜂屬於防禦型個股
# 如大盤為上漲循環，持有卜蜂的報酬率會低於大盤；
# 如大盤為下跌循環，卜蜂下跌的幅度，會小於大盤下跌的幅度。
# 持有卜蜂一年，期預期報酬率為7.21%。
# 預測一年後卜蜂的股價為每股NT$119.00元。
#===================================================================================


















