from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 博文：http://www.cnblogs.com/Lin-Yi/p/8972051.html

'''
随机森林回归
极端随机森林回归
梯度提升回归

通常集成模型能够取得非常好的表现
'''

# 1 准备数据
# 读取波士顿地区房价信息
boston = load_boston()
# 查看数据描述
# print(boston.DESCR)   # 共506条波士顿地区房价信息，每条13项数值特征描述和目标房价
# 查看数据的差异情况
# print("最大房价：", np.max(boston.target))   # 50
# print("最小房价：",np.min(boston.target))    # 5
# print("平均房价：", np.mean(boston.target))   # 22.532806324110677

x = boston.data
y = boston.target

# 2 分割训练数据和测试数据
# 随机采样25%作为测试 75%作为训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 3 训练数据和测试数据进行标准化处理
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 4 三种集成回归模型进行训练和预测
# 随机森林回归
rfr = RandomForestRegressor()
# 训练
rfr.fit(x_train, y_train)
# 预测 保存预测结果
rfr_y_predict = rfr.predict(x_test)

# 极端随机森林回归
etr = ExtraTreesRegressor()
# 训练
etr.fit(x_train, y_train)
# 预测 保存预测结果
etr_y_predict = rfr.predict(x_test)

# 梯度提升回归
gbr = GradientBoostingRegressor()
# 训练
gbr.fit(x_train, y_train)
# 预测 保存预测结果
gbr_y_predict = rfr.predict(x_test)

# 5 模型评估
# 随机森林回归模型评估
print("随机森林回归的默认评估值为：", rfr.score(x_test, y_test))
print("随机森林回归的R_squared值为：", r2_score(y_test, rfr_y_predict))
print("随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                          ss_y.inverse_transform(rfr_y_predict)))
print("随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(rfr_y_predict)))

# 极端随机森林回归模型评估
print("极端随机森林回归的默认评估值为：", etr.score(x_test, y_test))
print("极端随机森林回归的R_squared值为：", r2_score(y_test, gbr_y_predict))
print("极端随机森林回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(gbr_y_predict)))
print("极端随机森林回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(gbr_y_predict)))

# 梯度提升回归模型评估
print("梯度提升回归回归的默认评估值为：", gbr.score(x_test, y_test))
print("梯度提升回归回归的R_squared值为：", r2_score(y_test, etr_y_predict))
print("梯度提升回归回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                            ss_y.inverse_transform(etr_y_predict)))
print("梯度提升回归回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(etr_y_predict)))

'''
随机森林回归的默认评估值为： 0.8391590262557747
随机森林回归的R_squared值为： 0.8391590262557747
随机森林回归的均方误差为: 12.471817322834646
随机森林回归的平均绝对误差为: 2.4255118110236227

极端随机森林回归的默认评估值为： 0.783339502805047
极端随机森林回归的R_squared值为： 0.8391590262557747
极端随机森林回归的均方误差为: 12.471817322834646
极端随机森林回归的平均绝对误差为: 2.4255118110236227

GradientBoostingRegressor回归的默认评估值为： 0.8431187344932869
GradientBoostingRegressor回归的R_squared值为： 0.8391590262557747
GradientBoostingRegressor回归的均方误差为: 12.471817322834646
GradientBoostingRegressor回归的平均绝对误差为: 2.4255118110236227
'''

