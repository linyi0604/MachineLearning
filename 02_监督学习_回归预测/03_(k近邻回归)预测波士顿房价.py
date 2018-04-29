from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 博文：http://www.cnblogs.com/Lin-Yi/p/8971947.html

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

# 4 两种k近邻回归行学习和预测
# 初始化k近邻回归模型 使用平均回归进行预测
uni_knr = KNeighborsRegressor(weights="uniform")
# 训练
uni_knr.fit(x_train, y_train)
# 预测 保存预测结果
uni_knr_y_predict = uni_knr.predict(x_test)

# 多初始化k近邻回归模型 使用距离加权回归
dis_knr = KNeighborsRegressor(weights="distance")
# 训练
dis_knr.fit(x_train, y_train)
# 预测 保存预测结果
dis_knr_y_predict = dis_knr.predict(x_test)

# 5 模型评估
# 平均k近邻回归 模型评估
print("平均k近邻回归的默认评估值为：", uni_knr.score(x_test, y_test))
print("平均k近邻回归的R_squared值为：", r2_score(y_test, uni_knr_y_predict))
print("平均k近邻回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(uni_knr_y_predict)))
print("平均k近邻回归 的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(uni_knr_y_predict)))
# 距离加权k近邻回归 模型评估
print("距离加权k近邻回归的默认评估值为：", dis_knr.score(x_test, y_test))
print("距离加权k近邻回归的R_squared值为：", r2_score(y_test, dis_knr_y_predict))
print("距离加权k近邻回归的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                             ss_y.inverse_transform(dis_knr_y_predict)))
print("距离加权k近邻回归的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                                ss_y.inverse_transform(dis_knr_y_predict)))

'''
平均k近邻回归的默认评估值为： 0.6903454564606561
平均k近邻回归的R_squared值为： 0.6903454564606561
平均k近邻回归的均方误差为: 24.01101417322835
平均k近邻回归 的平均绝对误差为: 2.9680314960629928
距离加权k近邻回归的默认评估值为： 0.7197589970156353
距离加权k近邻回归的R_squared值为： 0.7197589970156353
距离加权k近邻回归的均方误差为: 21.730250160926044
距离加权k近邻回归的平均绝对误差为: 2.8050568785108005
'''

