import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
import numpy as np
import pylab as pl

# 博文： http://www.cnblogs.com/Lin-Yi/p/8974439.html

'''
特征提取：
    特征降维的手段
    抛弃对结果没有联系的特征
    抛弃对结果联系较少的特征
    以这种方式，降低维度
    
数据集的特征过多，有些对结果没有任何关系，
这个时候，将没有关系的特征删除，反而能获得更好的预测结果

下面使用决策树，预测泰坦尼克号幸存情况，
对不同百分比的筛选特征，进行学习和预测，比较准确率
'''

# 1 准备数据
titanic = pd.read_csv("../data/titanic/titanic.txt")
# 分离数据特征与目标
y = titanic["survived"]
x = titanic.drop(["row.names", "name", "survived"], axis=1)
# 对缺失值进行补充
x['age'].fillna(x['age'].mean(), inplace=True)
x.fillna("UNKNOWN", inplace=True)

# 2 分割数据集 25%用于测试 75%用于训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

# 3 类别型特征向量化
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))
# 输出处理后向量的维度
# print(len(vec.feature_names_))  # 474

# 4 使用决策树对所有特征进行学习和预测
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train, y_train)
print("全部维度的预测准确率：", dt.score(x_test, y_test))  # 0.8206686930091185

# 5 筛选前20%的特征，使用相同配置的决策树模型进行评估性能
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
x_train_fs = fs.fit_transform(x_train, y_train)
x_test_fs = fs.transform(x_test)
dt.fit(x_train_fs, y_train)
print("前20%特征的学习模型预测准确率：", dt.score(x_test_fs, y_test))     # 0.8237082066869301

# 6 通过交叉验证 按照固定间隔百分比筛选特征， 展示性能情况
percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    x_train_fs = fs.fit_transform(x_train, y_train)
    scores = cross_val_score(dt, x_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
# print(results)
'''
[0.85063904 0.85673057 0.87501546 0.88622964 0.86284271 0.86489384
 0.87303649 0.86689342 0.87098536 0.86690373 0.86895485 0.86083282
 0.86691404 0.86488353 0.86895485 0.86792414 0.86284271 0.86995465
 0.86486291 0.86385281 0.86384251 0.86894455 0.86794475 0.86690373
 0.86488353 0.86489384 0.86590394 0.87300557 0.86995465 0.86793445
 0.87097506 0.86998557 0.86692435 0.86892393 0.86997526 0.87098536
 0.87198516 0.86691404 0.86691404 0.87301587 0.87202639 0.8648423
 0.86386312 0.86388374 0.86794475 0.8618223  0.85877139 0.86285302
 0.86692435 0.8577819 ]
'''
# 找到最佳性能的筛选百分比
opt = np.where(results == results.max())[0][0]
print("最高性能的筛选百分比是：%s%%" % percentiles[opt])  # 7

pl.plot(percentiles, results)
pl.xlabel("percentiles of features")
pl.ylabel("accuracy")
pl.show()

