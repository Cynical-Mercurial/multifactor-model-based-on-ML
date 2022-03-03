from atrader import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import datetime as dt
from cvxopt import solvers, matrix
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 初始化
def init(context):
    set_backtest(initial_cash=10000000)        # 设置回测初始信息
    reg_kdata('month', 1)                        # 注册K线数据
    context.name = ['obv20', 'rev_grow_f3',
                    'roe', 'roa',
                    'operating_cashflow_ps', 'gross_income_ratio',
                    'pb', 'operating_profit_ps',
                    'basic_eps', 'momentum12m']
    reg_factor(context.name)                   # 注册因子数据
    context.factor_direction = [-1,1,1,1,1,1,1,1,1,1]     # 1表示正向因子，-1表示负向因子,必须与上面的因子数量一致
    context.date = 3
    context.number = 6                         # 12期的因子数据
    context.factor_return = pd.DataFrame()     # 预期因子收益率
    context.specific_return = pd.DataFrame()
    context.factor_value = pd.DataFrame()      # 保存上一期的因子
    context.stock_number = 50
    context.landa = 2                          # 风险厌恶系数

# 策略逻辑运算实现
def on_data(context):
    # 因子数据
    factors = get_reg_factor(context.reg_factor[0], target_indices=[], length=1, df=True)
    factor = factors.pivot_table(values='value', index='target_idx', columns='factor', aggfunc='min')
    factor = factor * context.factor_direction
    
    # 行情数据
    kdata = get_reg_kdata(context.reg_kdata[0], target_indices=[],length=context.date,fill_up=True,df=True)
    if kdata['close'].isna().any():                       # 如果数据不满21天则跳过
        context.factor_value = factor
        return
    
    # 求取股票收益率
    close = kdata.pivot_table(values='close',index='target_idx',columns='time',aggfunc='min')
    stock_return = close.iloc[:, -1].div(close.iloc[:, -2]) - 1         # 当期股票收益率
    stock_return[np.isinf(stock_return)] = np.nan                       # inf值替换为nan

    # 因子数据和行情数据合并
    alldata = pd.concat([context.factor_value,stock_return],axis=1)                   # 上一期的因子和当期股票收益率合并
   
    # 去NaN值,去停牌股票
    # alldata.dropna(axis=0, how='any',inplace=True)
    alldata.fillna(0,inplace=True)

    # 因子标准化、去极值
    factor_test = alldata[context.name]                                 # 需要处理的因子值
    factor_process = standardize_z(extreme_MAD(factor_test))            # 取极值，标准化，不用考虑行索引
    factor_process.fillna(0,inplace=True)
    
    # 多元回归得到因子收益率
    result = sm.OLS(alldata.iloc[:,-1].astype(float),factor_process.astype(float)).fit()
    factor_return_one = result.params                                        # 上一期的因子收益率
    specific_return_one = result.resid                                       # 上一期的特异性收益
    context.factor_return.insert(0,context.now,factor_return_one)            # 将上一期的因子收益率放入dataframe中
    context.specific_return.insert(0, context.now, specific_return_one)      # 将上一期的特异性收益率放入dataframe中
    
    # 保证获取6期的因子值
    if context.factor_return.T.shape[0] < context.number:                    # 获取n期的值
        return
    
     # 各期因子的因子收益率值 f
    factor_return = context.factor_return.T.iloc[:context.number,:]

    ### 股票收益率预测
    # 因子收益率预测
    factor_return = factor_return.iloc[::-1,:]
    # 滑动加权平均法预测本期因子收益率
    pre_factor_return = factor_return.ewm(span=2, ignore_na=False, adjust=False,axis=0).mean().iloc[-1]     # 本期因子收益率的预测
    # 本期因子值
    factor_today = standardize_z(extreme_MAD(factor.fillna(0)))
    # 预测股票下一期的收益率
    pre_stock_return = factor_today.dot(pre_factor_return)
    # 选择排名前n支股票进入股票池
    stock_rank = pre_stock_return.sort_values(ascending=False).head(context.stock_number)
    symbols_pool = stock_rank.index.tolist()

    ### 股票的方差协方差矩阵预测
    # 因子收益协方差矩阵（K*K）
    factor_covaraicne = np.cov(factor_return,rowvar=False)
    # 股票特有风险,股票的特质波动率矩阵（N*N）
    specific_return = specific_return_one[symbols_pool]     # 获取股票池中标的特有风险
    specific_risk = np.diagflat(list(specific_return))
    # 因子共同方差协方差矩阵 （N*N）
    factor_today_pool = factor_today.loc[symbols_pool]
    F = np.dot(np.dot(factor_today_pool,factor_covaraicne),factor_today_pool.T)
    # 股票的方差协方差矩阵
    V = F + specific_risk

    ### 组合优化，二次规划
    # 目标函数，最小化风险和最大收益
    P = matrix(context.landa * V)         # matrix里区分int和double，所以数字后面都需要加小数点
    q = matrix(-stock_rank)

    # 上下限的约束
    G = matrix(np.vstack((-1 * np.eye(len(V)),np.eye(len(V)))))
    a = list(np.linspace(0.0, 0.0, len(V)))
    b = list(np.linspace(1.0, 1.0, len(V)))
    c = a + b
    h = matrix(c)

    # 整体权重为1
    A = matrix(np.ones(len(V))).T
    b = matrix([1.00])
    sol = solvers.qp(P, q, G, h, A, b)   # 组合的权重

    percent = sol['x']       # 每支股票的仓位

    # 仓位数据查询
    positions = context.account().positions

    # 策略下单交易：
    # 平不在标的池的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in symbols_pool:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_target_volume(account_idx=0, target_idx=target_idx, target_volume=0, side=1, order_type=2,
                                    price=0)

    # 买在标的池中的股票
    for i in range(len(symbols_pool)):
        order_target_percent(account_idx=0, target_idx=int(symbols_pool[i]), target_percent=percent[i], side=1, order_type=2,
                             price=0)

if __name__ == '__main__':
    begin = '2018-01-01'
    end = '2021-03-31'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    hs300 = get_code_list('hs300', end)
    run_backtest(strategy_name='组合优化多因子选股',
                 file_path='.',
                 target_list=list(hs300['code']),
                 frequency='month',
                 fre_num=1,
                 begin_date=begin,
                 end_date=end,
                 fq=1)


# MAD:中位数去极值
def extreme_MAD(dt,n = 5.2):
    median = dt.quantile(0.5)   # 找出中位数
    new_median = (abs((dt - median)).quantile(0.5))   # 偏差值的中位数
    dt_up = median + n*new_median    # 上限
    dt_down = median - n*new_median  # 下限
    return dt.clip(dt_down, dt_up, axis=1)    # 超出上下限的值，赋值为上下限

# Z值标准化
def standardize_z(dt):
    mean = dt.mean()     #  截面数据均值
    std = dt.std()       #  截面数据标准差
    return (dt - mean)/std


