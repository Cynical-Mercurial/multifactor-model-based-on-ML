# multifactor-model-based-on-ML
pyhon: portfolio based on randomforest


一、策略思路
首先根据自己的喜好人工选择因子，或者基于统计数据选择因子，并用单因子分析检验因子有效性以及因子方向（IC值法）。我们将采用机器学习的算法对股票进行选股买入。机器学习多因子选股认为因子值和股票收益率存在一定的数量关系，使用机器学习建立模型： 
数据集：历史前n个月的因子值和股票收益率 
特征：前n期至上一期的股票因子值 
标签：前n-1期至本期的股票收益率 使用机器学习模型，通过本期的因子值数据，预测下一期股票的收益率，选择预期收益率大于0的 股票买入。


二、因子分析（IC值法验证因子有效性，以及判定因子方向）
以OBV20(20日能量潮指标), rev_grow_f3预期营收增长率（三年）为例。
IC值（Information Coefficient 信息系数）代表因子预测股票收益的能力无论使用 Normal IC 还是 Rank IC，IC值的绝对值都越大越好，越大说明因子选股的能力越强。
	IC>0，则因子为正向因子；IC<0，则因子为反向因子；
	通常 IC 大于2%或者小于-2%，则认为因子比较有效。

三、绩效分析
累计收益率为48.65%，夏普比率为0.59，获得正阿尔法收益，净值曲线高于沪深300基准收益率，说明随机森林多因子选股的可行性。
多因子选股常被用于量化分析，但针对多因子模型本身，还有一些值得我去总结思考的方面。通过对系统性风险的学习，我们知道系统风险并不仅仅只是整个市场或经济的波动，还可能来在其他源头，这些产生于其他源头的系统风险再回在期望回报率中形成风险溢价。因此，我们看到某个投资者获得了正的阿尔法时，有可能确实是因为她投资能力强，也可能是因为他其实只是承担了一些我们还未观测到的系统性风险。而且在因子拟合模型中，用当前的因子来解释当前资产收益率的差异，往往可以得到相当不错的拟合优度。但在预测模型中，即用当前因子预测未来资产收益率，相关系数就很低，甚至很难超过2%(IC值>2%或<-2%的因子被认定为有效因子)诸多的经验已经表明了，在真实世界中预测资产回报率是相当困难的。因此，不能被因子拟合模型的高拟合优度和回测绩效所误导，而对利用因子模型来进行投资抱有过高期待。
