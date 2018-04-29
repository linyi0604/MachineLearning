from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 博文：http://www.cnblogs.com/Lin-Yi/p/8972002.html

'''
回归树：
    严格上说 回归树不能算是回归
    叶子节点是一团训练数据的均值 不是连续 具体的预测值
    
    解决特征非线性的问题
    不要求特征标准化和统一量化
    
    容易过于复杂丧失泛化能力
    稳定性较差，细微改变会导致树结构发生重大变化
    
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

# 4 使用回归树进行训练和预测
# 初始化k近邻回归模型 使用平均回归进行预测
dtr = DecisionTreeRegressor()
# 训练
dtr.fit(x_train, y_train)
# 预测 保存预测结果
dtr_y_predict = dtr.predict(x_test)

# 5 模型评估
print("回归树的默认评估值为：", dtr.score(x_test, y_test))
print("平回归树的R_squared值为：", r2_score(y_test, dtr_y_predict))
print("回归树的均方误差为:", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(dtr_y_predict)))
print("回归树的平均绝对误差为:", mean_absolute_error(ss_y.inverse_transform(y_test),
                                               ss_y.inverse_transform(dtr_y_predict)))

'''
回归树的默认评估值为： 0.7066505912533438
平回归树的R_squared值为： 0.7066505912533438
回归树的均方误差为: 22.746692913385836
回归树的平均绝对误差为: 3.08740157480315
'''

