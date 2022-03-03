from atrader import *
import numpy as np
import pandas as pd
import math
import statsmodels.api as sm
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


set_setting('ALLOW_CONSOLE_SYSTEM_WARN',False)

# 初始化
def init(context):
    set_backtest(initial_cash=100000000)        # 设置回测初始信息
    reg_kdata('month', 1)                        # 注册K线数据
    context.name = ['obv20', 'rev_grow_f3',
                    'roe', 'roa',
                    'operating_cashflow_ps', 'gross_income_ratio',
                    'pb','operating_profit_ps',
                    'basic_eps','momentum12m']
    reg_factor(context.name)                   # 注册因子数据
    context.factor_direction = [-1,1,1,1,1,1,1,1,1,1]  # 与因子名字对应，正向因子为1，反向因子为-1
    context.date = 3                           # 注册K线数据长度
    context.d = 0
    context.number = []                        # 记录上一期股票的数量
    context.n = 3                              # 开始回测的因子数据月份长度
    context.factordata = pd.DataFrame()        # 因子上期数据
    context.stock_data = pd.DataFrame()        # 收益率历史数据
    context.factors_data = pd.DataFrame()      # 因子历史数据


# 策略逻辑运算实现
def on_data(context):


    # 因子数据
    factors = get_reg_factor(context.reg_factor[0], target_indices=[], length=1, df=True)
    factor = factors.pivot_table(values='value', index='target_idx', columns='factor', aggfunc='min')
    # 全部转为正向因子
    factor_data  = factor * context.factor_direction

    # 行情数据
    kdata = get_reg_kdata(context.reg_kdata[0], target_indices=[],length=context.date,fill_up=True,df=True)
    if kdata['close'].isna().any():                                     # 如果数据不满注册数据长度则跳过
        context.factordata = factor                                     # 因子数据赋值给全局变量
        return
    close = kdata.pivot_table(values='close',index='target_idx',columns='time',aggfunc='min')
    stock_return = close.iloc[:, -1].div(close.iloc[:, -2]) - 1         # 当期股票收益率
    stock_return[np.isinf(stock_return)] = np.nan                       # inf值替换为nan

    # 因子数据和行情数据合并
    alldata = pd.concat([context.factordata,stock_return],axis=1)       # 上一期的因子和当期股票收益率合并
    context.factordata = factor                                         # 当期的因子值赋值给全局变量factordata
    # 去NaN值,去停牌股票
    alldata.dropna(axis=0, how='any',inplace=True)


    # 因子标准化、去极值、中性化
    factor_test = alldata[context.name]                                   # 需要处理的因子值
    factor_process = standardize_z(extreme_MAD(factor_test))              # 取极值，标准化，不用考虑行索引
  

    ## 策略逻辑

    # 收益率标签处理
    s = [context.stock_data,alldata.iloc[:,-1]]
    context.stock_data = pd.concat(s)

    # 因子特征集处理
    num = factor_process.shape[0]                              # 记录数据长度，以便删减
    context.number.append(num)                                 # 记录数据长度，以便删减
    frames = [context.factors_data,factor_process]
    context.factors_data = pd.concat(frames)                   # 将上一期的因子值放入dataframe中
    context.d = 1 + context.d                                  # 记录因子的期数
    if context.d < context.n:                                  # 当数据小于n期时,结束跳出
        return
    elif context.d > context.n:
        context.factors_data = context.factors_data.iloc[context.number[-context.n-1]:,:]
        context.stock_data = context.stock_data.iloc[context.number[-context.n-1]:,:]

    # 数据集划分
    context.stock_data[context.stock_data > 0] = 1        # 收益率大于0，设为1
    context.stock_data[context.stock_data <= 0] = 0
    X_train, X_test, y_train, y_test = train_test_split(context.factors_data, context.stock_data, test_size=0.1, random_state=2)

    # 模型训练
    clf=RandomForestClassifier(max_features='log2',n_estimators=10,bootstrap=True,max_depth=3) 
    clf.fit(X_train,np.asarray(y_train).ravel())
    # 模型得分
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('train_score: ',train_score)
    print('test_score: ', test_score)
    # 测试集得分
    Y_pre = clf.predict(X_test)
    recall = metrics.recall_score(Y_pre, y_test)
    f1 = metrics.f1_score(Y_pre, y_test)
    print('recall: ',recall)
    print('f1: ', f1)
    clf.fit(context.factors_data, np.asarray(context.stock_data).ravel())

    # 当期因子值矩阵处理
    now_factor_test = factor[context.name]                                   # 当期需要处理的因子值
    now_factor_test2 = standardize_z(extreme_MAD(now_factor_test))           # 取极值，标准化，不用考虑行索引
    now_factor_test2.fillna(0,inplace=True)

    # 得到预测的股票收益率
    pre_stock_return = clf.predict(now_factor_test2)
    pre_stock_return2 = pd.DataFrame(pre_stock_return,index = now_factor_test2.index,columns=['data'])
    stock_list = pre_stock_return2.loc[pre_stock_return2['data'] == 1]                              # 取模型预测为1的股票
    # stock_list = pre_stock_return2.sort_values(by =['data'],axis=0,ascending=False).head(30)      # 降序，取前30支股票

    # 建立股票池
    symbols_pool = stock_list.index.tolist()
    positions = context.account().positions            #  仓位情况


    # 平不在标的池的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in symbols_pool:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_target_volume(account_idx=0, target_idx=int(target_idx), target_volume=0, side=1, order_type=2,
                                    price=0)
                # print('市价单平不在标的池的', context.target_list[target_idx])

    if len(symbols_pool) == 0:
        return

    # 获取股票的权重
    percent = 0.9 / len(symbols_pool)

    # 买在标的池中的股票
    for target_idx in symbols_pool:
        order_target_percent(account_idx=0, target_idx=int(target_idx), target_percent=percent, side=1, order_type=2,
                             price=0)
        # print(context.target_list[int(target_idx)], '以市价单调多仓到仓位', percent)
    
    
    
    
    
if __name__ == '__main__':
    begin = '2018-01-01'
    end = '2021-04-01'
    cons_date = dt.datetime.strptime(begin, '%Y-%m-%d') - dt.timedelta(days=1)
    hs300 = get_code_list('hs300', cons_date)
    run_backtest(strategy_name='基于机器学习多因子选股模型',
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
    mean = dt.mean(axis = 0)     #  截面数据均值
    std = dt.std(axis = 0)       #  截面数据标准差
    return (dt - mean)/std

