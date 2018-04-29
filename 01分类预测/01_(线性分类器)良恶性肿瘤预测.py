import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report

'''
线性分类器
最基本和常用的机器学习模型
受限于数据特征与分类目标的线性假设
逻辑斯蒂回归 计算时间长，模型性能略高
随机参数估计 计算时间短，模型性能略低
'''

'''
1 数据预处理
'''
# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell size',
                'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# 使用pandas.read_csv取数据集
data = pd.read_csv('./data/breast/breast-cancer-wisconsin.data', names=column_names)
# 将?替换为标准缺失值表示
data = data.replace(to_replace='?', value=np.nan)
# 丢失带有缺失值的数据 只要有一个维度有缺失就丢弃
data = data.dropna(how='any')
# 输出data数据的数量和维度
# print(data.shape)


'''
2 准备 良恶性肿瘤训练、测试数据部分
'''
# 随机采样25%数据用于测试 75%数据用于训练
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]],
                                                    data[column_names[10]],
                                                    test_size=0.25,
                                                    random_state=33)
# 查验训练样本和测试样本的数量和类别分布
# print(y_train.value_counts())
# print(y_test.value_counts())
'''
训练样本共512条 其中344条良性肿瘤  168条恶性肿瘤
2    344
4    168
Name: Class, dtype: int64
测试数据共171条 其中100条良性肿瘤 71条恶性肿瘤
2    100
4     71
Name: Class, dtype: int64
'''


'''
3 机器学习模型进行预测部分
'''
# 数据标准化，保证每个维度特征的方差为1 均值为0 预测结果不会被某些维度过大的特征值主导
ss = StandardScaler()
x_train = ss.fit_transform(x_train)     # 对x_train进行标准化
x_test = ss.transform(x_test)       # 用与x_train相同的规则对x_test进行标准化，不重新建立规则

# 分别使用 逻辑斯蒂回归 和 随机参数估计 两种方法进行学习预测

lr = LogisticRegression()   # 初始化逻辑斯蒂回归模型
sgdc = SGDClassifier()  # 初始化随机参数估计模型

# 使用 逻辑斯蒂回归 在训练集合上训练
lr.fit(x_train, y_train)
# 训练好后 对测试集合进行预测 预测结果保存在 lr_y_predict中
lr_y_predict = lr.predict(x_test)

# 使用 随机参数估计 在训练集合上训练
sgdc.fit(x_train, y_train)
# 训练好后 对测试集合进行预测 结果保存在 sgdc_y_predict中
sgdc_y_predict = sgdc.predict(x_test)

'''
4 性能分析部分
'''
# 逻辑斯蒂回归模型自带评分函数score获得模型在测试集合上的准确率
print("逻辑斯蒂回归准确率：", lr.score(x_test, y_test))
# 逻辑斯蒂回归的其他指标
print("逻辑斯蒂回归的其他指标：\n", classification_report(y_test, lr_y_predict, target_names=["Benign", "Malignant"]))

# 随机参数估计的性能分析
print("随机参数估计准确率：", sgdc.score(x_test, y_test))
# 随机参数估计的其他指标
print("随机参数估计的其他指标:\n", classification_report(y_test, sgdc_y_predict, target_names=["Benign", "Malignant"]))

'''
recall 召回率
precision 精确率
fl-score
support

逻辑斯蒂回归准确率： 0.9707602339181286
逻辑斯蒂回归的其他指标：
              precision    recall  f1-score   support

     Benign       0.96      0.99      0.98       100
  Malignant       0.99      0.94      0.96        71

avg / total       0.97      0.97      0.97       171

随机参数估计准确率： 0.9649122807017544
随机参数估计的其他指标:
              precision    recall  f1-score   support

     Benign       0.97      0.97      0.97       100
  Malignant       0.96      0.96      0.96        71

avg / total       0.96      0.96      0.96       171
'''