from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
# 导入多项式特征生成器
from sklearn.preprocessing import PolynomialFeatures

# 查看图片可以去博客：http://www.cnblogs.com/Lin-Yi/p/8975638.html

'''
在做线性回归预测时候，
为了提高模型的泛化能力，经常采用多次线性函数建立模型

f = k*x + b   一次函数
f = a*x^2 + b*x + w  二次函数
f = a*x^3 + b*x^2 + c*x + w  三次函数
。。。

泛化：
    对未训练过的数据样本进行预测。

欠拟合:
    由于对训练样本的拟合程度不够，导致模型的泛化能力不足。

过拟合：
    训练样本拟合非常好，并且学习到了不希望学习到的特征，导致模型的泛化能力不足。


在建立超过一次函数的线性回归模型之前，要对默认特征生成多项式特征再输入给模型

下面模拟 根据蛋糕的直径大小 预测蛋糕价格

'''

# 样本的训练数据，特征和目标值
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 一次线性回归的学习与预测
# 线性回归模型 学习
# 进行四次线性回归模型拟合
poly4 = PolynomialFeatures(degree=4)  # 4次多项式特征生成器
x_train_poly4 = poly4.fit_transform(x_train)
# 建立模型预测
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
print("四次线性模型预测得分:", regressor_poly4.score(x_train_poly4, y_train))  # 1.0
