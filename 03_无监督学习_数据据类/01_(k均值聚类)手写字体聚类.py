import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

# 博文： http://www.cnblogs.com/Lin-Yi/p/8972996.html
'''
k均值算法：
    1 随机选择k个样本作为k个类别的中心
    2 从k个样本出发，选取最近的样本归为和自己同一个分类，一直到所有样本都有分类
    3 对k个分类重新计算中心样本
    4 从k个新中心样本出发重复23，
        如果据类结果和上一次一样，则停止
        否则重复234
        
'''
'''
该数据集源自网上 https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/
我把他下载到了本地
训练样本3823条， 测试样本1797条
图像通过8*8像素矩阵表示共64个维度，1个目标维度表示数字类别
'''

# 1 准备数据
digits_train = pd.read_csv("./data/optdigits/optdigits.tra", header=None)
digits_test = pd.read_csv("./data/optdigits/optdigits.tes", header=None)
# 从样本中抽取出64维度像素特征和1维度目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 2 建立模型
# 初始化kMeans聚类模型 聚类中心数量为10个
kmeans = KMeans(n_clusters=10)
# 聚类
kmeans.fit(x_train)
# 逐条判断每个测试图像所属的聚类中心你
y_predict = kmeans.predict(x_test)


# 3 模型评估
# 使用ARI进行性能评估 当聚类有所属类别的时候利用ARI进行模型评估
print("k均值聚类的ARI值：", metrics.adjusted_rand_score(y_test, y_predict))
'''
k均值聚类的ARI值： 0.6673881543921809
'''
# 如果没有聚类所属类别，利用轮廓系数进行评估
