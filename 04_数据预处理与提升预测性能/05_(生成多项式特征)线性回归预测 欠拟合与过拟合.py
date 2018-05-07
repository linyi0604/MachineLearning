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
regressor = LinearRegression()
regressor.fit(x_train, y_train)
# 画出一次线性回归的拟合曲线
xx = np.linspace(0, 25, 100)   # 0到16均匀采集100个点做x轴
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)  # 计算每个点对应的y
plt.scatter(x_train, y_train)   # 画出训练数据的点
plt1, = plt.plot(xx, yy, label="degree=1")
plt.axis([0, 25, 0, 25])
plt.xlabel("Diameter")
plt.ylabel("Price")
plt.legend(handles=[plt1])
plt.show()
# 输出在样本上的预测评分
print("一次线性模型在预测集合上得分：", regressor.score(x_train, y_train))     # 0.9100015964240102


# 2次线性回归进行预测
poly2 = PolynomialFeatures(degree=2)    # 2次多项式特征生成器
x_train_poly2 = poly2.fit_transform(x_train)
# 建立模型预测
regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2, y_train)
# 画出2次线性回归的图
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label="Degree1")
plt2, = plt.plot(xx, yy_poly2, label="Degree2")
plt.axis([0, 25, 0, 25])
plt.xlabel("Diameter")
plt.ylabel("Price")
plt.legend(handles=[plt1, plt2])
plt.show()
# 输出二次回归模型的预测样本评分
print("二次线性模型在预测集合上得分:", regressor_poly2.score(x_train_poly2, y_train))     # 0.9816421639597427


# 进行四次线性回归模型拟合
poly4 = PolynomialFeatures(degree=4)    # 4次多项式特征生成器
x_train_poly4 = poly4.fit_transform(x_train)
# 建立模型预测
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
# 画出2次线性回归的图
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
plt.scatter(x_train, y_train)
plt1, = plt.plot(xx, yy, label="Degree1")
plt2, = plt.plot(xx, yy_poly2, label="Degree2")
plt4, = plt.plot(xx, yy_poly4, label="Degree2")
plt.axis([0, 25, 0, 25])
plt.xlabel("Diameter")
plt.ylabel("Price")
plt.legend(handles=[plt1, plt2, plt4])
plt.show()
# 输出二次回归模型的预测样本评分
print("四次线性模型在预测集合上得分:", regressor_poly4.score(x_train_poly4, y_train))     # 1.0


# 准备测试数据
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
print("一次线性模型在测试集合上得分:", regressor.score(x_test, y_test))   # 0.809726797707665
x_test_poly2 = poly2.transform(x_test)
print("二次线性模型在测试集合上得分:", regressor_poly2.score(x_test_poly2, y_test))   # 0.8675443656345054
x_test_poly4 = poly4.transform(x_test)
print("四次线性模型在测试集合上得分:", regressor_poly4.score(x_test_poly4, y_test))   # 0.8095880795746723
