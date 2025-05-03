import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 設置支持中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微軟雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 定義交易策略和回測邏輯
def sma_crossover_strategy(data, short_window, long_window):
    # 計算移動平均線
    data['Short_SMA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_SMA'] = data['Close'].rolling(window=long_window).mean()
    
    # 初始化信號和持倉列
    data['Signal'] = 0
    data['Position'] = 0
    
    # 生成交易信號
    for i in range(1, len(data)):
        if pd.notna(data['Short_SMA'].iloc[i]) and pd.notna(data['Long_SMA'].iloc[i]):
            if data['Short_SMA'].iloc[i] > data['Long_SMA'].iloc[i] and data['Short_SMA'].iloc[i-1] <= data['Long_SMA'].iloc[i-1]:
                data.loc[data.index[i], 'Signal'] = 1  # 買入信號
            elif data['Short_SMA'].iloc[i] < data['Long_SMA'].iloc[i] and data['Short_SMA'].iloc[i-1] >= data['Long_SMA'].iloc[i-1]:
                data.loc[data.index[i], 'Signal'] = -1  # 賣出信號
    
    # 計算持倉（1 為持有，0 為空倉）
    position = 0  # 初始持倉為 0
    positions = [0] * len(data)  # 創建持倉列表
    for i in range(len(data)):
        if data['Signal'].iloc[i] == 1:
            position = 1  # 買入後持倉為 1
        elif data['Signal'].iloc[i] == -1:
            position = 0  # 賣出後持倉為 0
        positions[i] = position
    data['Position'] = positions
    
    return data

def backtest(data):
    # 計算日回報
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)
    
    # 確保策略回報中的 NaN 被填充為 0
    data['Strategy_Returns'] = data['Strategy_Returns'].fillna(0)
    
    # 計算累積回報
    data['Cumulative_Returns'] = (1 + data['Returns']).cumprod() - 1
    data['Strategy_Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod() - 1
    
    # 計算總回報和年化回報
    total_return = data['Strategy_Cumulative_Returns'].iloc[-1]
    days = (data.index[-1] - data.index[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
    
    # 計算夏普比率（假設無風險利率為 0）
    strategy_mean = data['Strategy_Returns'].mean() * 252
    strategy_std = data['Strategy_Returns'].std() * np.sqrt(252)
    sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0
    
    return data, total_return, annualized_return, sharpe_ratio

# 主程式
def main():
    # 下載歷史數據，顯式設置 auto_adjust=False
    ticker = 'AAPL'
    data = yf.download(ticker, start='2020-01-01', end='2025-04-30', auto_adjust=False)
    
    # 應用策略
    short_window = 20
    long_window = 50
    data = sma_crossover_strategy(data, short_window, long_window)
    
    # 執行回測
    data, total_return, annualized_return, sharpe_ratio = backtest(data)
    
    # 輸出結果
    print(f"總回報: {total_return:.2%}")
    print(f"年化回報: {annualized_return:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    
    # 可視化結果
    plt.figure(figsize=(12, 6))
    plt.plot(data['Cumulative_Returns'], label='買入並持有', color='blue', linewidth=2)
    plt.plot(data['Strategy_Cumulative_Returns'], label='策略', color='orange', linewidth=2)
    
    # 標記買賣信號
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['Cumulative_Returns'], marker='^', color='green', label='買入信號', s=100)
    plt.scatter(sell_signals.index, sell_signals['Cumulative_Returns'], marker='v', color='red', label='賣出信號', s=100)
    
    plt.title(f'{ticker} SMA 交叉策略回測')
    plt.xlabel('日期')
    plt.ylabel('累積回報')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_result.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()