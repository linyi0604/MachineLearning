from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 博文：http://www.cnblogs.com/Lin-Yi/p/8971798.html

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

# 4 使用两种线性回归模型进行训练和预测
# 初始化LinearRegression线性回归模型
lr = LinearRegression()
# 训练
lr.fit(x_train, y_train)
# 预测 保存预测结果
lr_y_predict = lr.predict(x_test)

# 初始化SGDRRegressor随机梯度回归模型
sgdr = SGDRegressor()
# 训练
sgdr.fit(x_train, y_train)
# 预测 保存预测结果
sgdr_y_predict = sgdr.predict(x_test)

# 5 模型评估
# 对Linear模型评估
lr_score = lr.score(x_test, y_test)
print("Linear的默认评估值为：", lr_score)
lr_R_squared = r2_score(y_test, lr_y_predict)
print("Linear的R_squared值为：", lr_R_squared)
lr_mse = mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print("Linear的均方误差为:", lr_mse)
lr_mae = mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print("Linear的平均绝对误差为:", lr_mae)

# 对SGD模型评估
sgdr_score = sgdr.score(x_test, y_test)
print("SGD的默认评估值为：", sgdr_score)
sgdr_R_squared = r2_score(y_test, sgdr_y_predict)
print("SGD的R_squared值为：", sgdr_R_squared)
sgdr_mse = mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))
print("SGD的均方误差为:", sgdr_mse)
sgdr_mae = mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))
print("SGD的平均绝对误差为:", sgdr_mae)

'''
Linear的默认评估值为： 0.6763403830998702
Linear的R_squared值为： 0.6763403830998701
Linear的均方误差为: 25.09698569206773
Linear的平均绝对误差为: 3.5261239963985433

SGD的默认评估值为： 0.659795654161198
SGD的R_squared值为： 0.659795654161198
SGD的均方误差为: 26.379885392159494
SGD的平均绝对误差为: 3.5094445431026413
'''

