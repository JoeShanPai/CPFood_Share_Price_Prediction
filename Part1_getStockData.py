# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:49:54 2025

@author: user
"""

"""
學號：B3871571
姓名：白若善
Python程式設計  自主學習 -- 專題製作
第一部份：由Yahoo Finance取得 0050.TW及 1215.TW的歷史股價資料
"""


#========================================================================
# 匯入所需模組及套件
#========================================================================
import yfinance as yf
import pandas as pd
import os


#=================================================================================================
# 定義函式，函式名為fetch_and_save_stock_data
# 函式內的重要參數，除了ticker外，就是確定歷史股價的起(start_date)、迄(end_date)日一致
# 抓取資料前，先直接在Yahoo Finance的網頁，觀察其html的架構
# 發現在歷史股價資訊的部份，採用了多層級索引，故決定用pandas將其扁平化處理
# 處理過後，只保留最頂層的欄位，例如："Date", "Open", "High".....
# Important I: Yahoo Finance雖然免費，但經常出現Rate Limited的訊息....
# Important II: 抓取失敗後，只能多等一下，再重新嘗試抓取。
#=================================================================================================
def fetch_and_save_stock_data(ticker: str, output_filename: str = "stock_data.csv", start_date="2008-01-02", end_date="2025-05-29"):
    print(f"正在從 Yahoo Finance 抓取 {ticker} 的歷史股價紀錄...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"警告：從 Yahoo Finance 抓取 {ticker} 的數據為空。請檢查股票代碼或稍後再試。")
            return # 如數據不存在，則直接退出函式

        if isinstance(data.columns, pd.MultiIndex):
            print("偵測到多層級欄位索引，正在將其扁平化。")
            data.columns = data.columns.droplevel(1)
            data = data.loc[:,~data.columns.duplicated()]

        data = data.reset_index()
        data.columns = data.columns.str.title()

        output_columns_order = ['Date', 'Open', 'High', 'Low', 'Close']
        # Yahoo Finance雖然免費，但裡面的資料非常混亂，只能多做幾次加工
        if 'Adj Close' not in data.columns:
            print("警告：'Adj Close' 欄位未在下載的數據中找到。'Close' 欄位已自動調整，將其作為 'Adj Close' 使用。")
            if 'Close' in data.columns:
                data['Adj Close'] = data['Close']
            else:
                print("錯誤：'Close' 欄位缺失，無法創建 'Adj Close'。請檢查 Yahoo Finance 數據源。")
                return
        
        output_columns_order.append('Adj Close')

        final_columns_to_select = [col for col in output_columns_order if col in data.columns]
        data_to_save = data[final_columns_to_select]

        if data_to_save.empty:
            print("警告：經過欄位選擇和處理後，要儲存的數據為空。")
            return
        # 設定相對路徑
        output_path = os.path.join(os.getcwd(), output_filename)
       
    except Exception as e:
        print(f"抓取或儲存數據時發生錯誤：{e}")

#========================================================================
# 請User自行輸入所需抓取資料的“股票代號”及“存檔檔名”
#========================================================================
stock_ticker = str(input("請輸入股票代號，例如：1101.TW： "))
output_csv_file = str(input("請輸入csv檔的檔名： "))

#========================================================================
# 呼叫函式並執行
#========================================================================
if __name__ == "__main__":
    fetch_and_save_stock_data(stock_ticker, output_csv_file)

#========================================================================
# 抓取ETF 0050(股票代號：0050.TW)的歷史股價資料
# 抓取卜蜂(股票代號：1215.TW)的歷史股價資料
# 將ETF 0050存為etf0059.csv檔
# 將卜蜂存為cpfood.csv檔 (卜蜂食品的英文為CP Food)
#========================================================================



