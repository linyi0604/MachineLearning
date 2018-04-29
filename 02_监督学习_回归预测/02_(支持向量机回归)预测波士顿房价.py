from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 博文：http://www.cnblogs.com/Lin-Yi/p/8971845.html

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

# 4.1 支持向量机模型进行学习和预测
# 线性核函数配置支持向量机
linear_svr = SVR(kernel="linear")
# 训练
linear_svr.fit(x_train, y_train)
# 预测 保存预测结果
linear_svr_y_predict = linear_svr.predict(x_test)

# 多项式核函数配置支持向量机
poly_svr = SVR(kernel="poly")
# 训练
poly_svr.fit(x_train, y_train)
# 预测 保存预测结果
poly_svr_y_predict = linear_svr.predict(x_test)

# 5 模型评估
# 线性核函数 模型评估
print("线性核函数支持向量机的默认评估值为：", linear_svr.score(x_test, y_test))
print("线性核函数支持向量机的R_squared值为：", r2_score(y_test, linear_svr_y_predict))
print("线性核函数支持向量机的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(linear_svr_y_predict)))
print("线性核函数支持向量机的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                                 ss_y.inverse_transform(linear_svr_y_predict)))
# 对多项式核函数模型评估
print("对多项式核函数的默认评估值为：", poly_svr.score(x_test, y_test))
print("对多项式核函数的R_squared值为：", r2_score(y_test, poly_svr_y_predict))
print("对多项式核函数的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(poly_svr_y_predict)))
print("对多项式核函数的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(poly_svr_y_predict)))

'''
线性核函数支持向量机的默认评估值为： 0.651717097429608
线性核函数支持向量机的R_squared值为： 0.651717097429608
线性核函数支持向量机的均方误差为: 27.0063071393243
线性核函数支持向量机的平均绝对误差为: 3.426672916872753
对多项式核函数的默认评估值为： 0.40445405800289286
对多项式核函数的R_squared值为： 0.651717097429608
对多项式核函数的均方误差为: 27.0063071393243
对多项式核函数的平均绝对误差为: 3.426672916872753
'''

