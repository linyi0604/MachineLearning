from sklearn.linear_model import LinearRegression, Lasso, Ridge
# 导入多项式特征生成器
from sklearn.preprocessing import PolynomialFeatures

# 博文： http://www.cnblogs.com/Lin-Yi/p/8999455.html

'''
正则化：
    提高模型在未知数据上的泛化能力
    避免参数过拟合
正则化常用的方法：
    在目标函数上增加对参数的惩罚项
    削减某一参数对结果的影响力度

L1正则化：lasso
    在线性回归的目标函数后面加上L1范数向量惩罚项。
    
    f = w * x^n + b + k * ||w||1 
    
    x为输入的样本特征
    w为学习到的每个特征的参数
    n为次数
    b为偏置、截距
    ||w||1 为 特征参数的L1范数，作为惩罚向量
    k 为惩罚的力度

L2范数正则化：ridge
    在线性回归的目标函数后面加上L2范数向量惩罚项。
    
    f = w * x^n + b + k * ||w||2 
    
    x为输入的样本特征
    w为学习到的每个特征的参数
    n为次数
    b为偏置、截距
    ||w||2 为 特征参数的L2范数，作为惩罚向量
    k 为惩罚的力度
        
        
下面模拟 根据蛋糕的直径大小 预测蛋糕价格
采用了4次线性模型，是一个过拟合的模型
分别使用两个正则化方法 进行学习和预测

'''

# 样本的训练数据，特征和目标值
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
# 准备测试数据
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
# 进行四次线性回归模型拟合
poly4 = PolynomialFeatures(degree=4)  # 4次多项式特征生成器
x_train_poly4 = poly4.fit_transform(x_train)
# 建立模型预测
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
x_test_poly4 = poly4.transform(x_test)
print("四次线性模型预测得分:", regressor_poly4.score(x_test_poly4, y_test))  # 0.8095880795746723

# 采用L1范数正则化线性模型进行学习和预测
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
print("L1正则化的预测得分为：", lasso_poly4.score(x_test_poly4, y_test))  # 0.8388926873604382

# 采用L2范数正则化线性模型进行学习和预测
ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print("L2正则化的预测得分为：", ridge_poly4.score(x_test_poly4, y_test))  # 0.8374201759366456
