from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import  PCA
import pandas as pd
import numpy as np

# 博文: http://www.cnblogs.com/Lin-Yi/p/8973077.html

'''
主成分分析：
    降低特征维度的方法。
    不会抛弃某一列特征，
    而是利用线性代数的计算，将某一维度特征投影到其他维度上去，
    尽量小的损失被投影的维度特征
    
    
api使用：
    estimator = PCA(n_components=20)
    pca_x_train = estimator.fit_transform(x_train)
    pca_x_test = estimator.transform(x_test)

分别使用支持向量机进行学习降维前后的数据再预测

该数据集源自网上 https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/
我把他下载到了本地
训练样本3823条， 测试样本1797条
图像通过8*8像素矩阵表示共64个维度，1个目标维度表示数字类别

'''

# 1 准备数据
digits_train = pd.read_csv("../data/optdigits/optdigits.tra", header=None)
digits_test = pd.read_csv("../data/optdigits/optdigits.tes", header=None)
# 从样本中抽取出64维度像素特征和1维度目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 2 对图像数据进行降维，64维度降低到20维度
estimator = PCA(n_components=20)
pca_x_train = estimator.fit_transform(x_train)
pca_x_test = estimator.transform(x_test)

# 3.1 使用默认配置的支持向量机进行学习和预测未降维的数据
svc = LinearSVC()
# 学习
svc.fit(x_train, y_train)
# 预测
y_predict = svc.predict(x_test)

# 3.2 使用默认配置的支持向量机学习和预测降维后的数据
pca_svc = LinearSVC()
# 学习
pca_svc.fit(pca_x_train, y_train)
pca_y_predict = pca_svc.predict(pca_x_test)

# 4 模型评估
print("原始数据的准确率：", svc.score(x_test, y_test))
print("其他评分：\n", classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

print("降维后的数据准确率:", pca_svc.score(pca_x_test, y_test))
print("其他评分：\n", classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))

'''
原始数据的准确率： 0.9165275459098498
其他评分：
              precision    recall  f1-score   support

          0       0.98      0.98      0.98       178
          1       0.73      0.99      0.84       182
          2       0.98      0.97      0.98       177
          3       0.96      0.88      0.92       183
          4       0.94      0.95      0.95       181
          5       0.91      0.96      0.93       182
          6       0.99      0.96      0.98       181
          7       0.98      0.92      0.95       179
          8       0.84      0.79      0.81       174
          9       0.94      0.76      0.84       180

avg / total       0.92      0.92      0.92      1797

降维后的数据准确率: 0.9220923761825265
其他评分：
              precision    recall  f1-score   support

          0       0.97      0.97      0.97       178
          1       0.93      0.86      0.89       182
          2       0.96      0.97      0.96       177
          3       0.93      0.87      0.90       183
          4       0.94      0.97      0.96       181
          5       0.86      0.96      0.91       182
          6       0.97      0.98      0.98       181
          7       0.97      0.88      0.92       179
          8       0.89      0.89      0.89       174
          9       0.82      0.88      0.85       180

avg / total       0.92      0.92      0.92      1797
'''