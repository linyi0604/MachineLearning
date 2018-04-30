import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

'''
博文：http://www.cnblogs.com/Lin-Yi/p/8971348.html

集成分类器：
综合考量多个分类器的预测结果做出考量。
这种综合考量大体上分两种：
    1 搭建多个独立的分类模型，然后通过投票的方式 比如 随机森林分类器
        随机森林在训练数据上同时搭建多棵决策树，这些决策树在构建的时候会放弃唯一算法，随机选取特征
    2 按照一定次序搭建多个分类模型，
        他们之间存在依赖关系，每一个后续模型的加入都需要现有模型的综合性能贡献，
        从多个较弱的分类器搭建出一个较为强大的分类器，比如梯度提升决策树
        提督森林决策树在建立的时候尽可能降低成体在拟合数据上的误差。
        
下面将对比 单一决策树 随机森林 梯度提升决策树 的预测情况

'''

'''
1 准备数据
'''
# 读取泰坦尼克乘客数据，已经从互联网下载到本地
titanic = pd.read_csv("../data/titanic/titanic.txt")
# 观察数据发现有缺失现象
# print(titanic.head())

# 提取关键特征，sex, age, pclass都很有可能影响是否幸免
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 查看当前选择的特征
# print(x.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 3 columns):
pclass    1313 non-null object
age       633 non-null float64
sex       1313 non-null object
dtypes: float64(1), object(2)
memory usage: 30.9+ KB
None
'''
# age数据列 只有633个，对于空缺的 采用平均数或者中位数进行补充 希望对模型影响小
x['age'].fillna(x['age'].mean(), inplace=True)

'''
2 数据分割
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
# 使用特征转换器进行特征抽取
vec = DictVectorizer()
# 类别型的数据会抽离出来 数据型的会保持不变
x_train = vec.fit_transform(x_train.to_dict(orient="record"))
# print(vec.feature_names_)   # ['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']
x_test = vec.transform(x_test.to_dict(orient="record"))

'''
3.1 单一决策树 训练模型 进行预测
'''
# 初始化决策树分类器
dtc = DecisionTreeClassifier()
# 训练
dtc.fit(x_train, y_train)
# 预测 保存结果
dtc_y_predict = dtc.predict(x_test)

'''
3.2 使用随机森林 训练模型 进行预测
'''
# 初始化随机森林分类器
rfc = RandomForestClassifier()
# 训练
rfc.fit(x_train, y_train)
# 预测
rfc_y_predict = rfc.predict(x_test)

'''
3.3 使用梯度提升决策树进行模型训练和预测
'''
# 初始化分类器
gbc = GradientBoostingClassifier()
# 训练
gbc.fit(x_train, y_train)
# 预测
gbc_y_predict = gbc.predict(x_test)


'''
4 模型评估
'''
print("单一决策树准确度:", dtc.score(x_test, y_test))
print("其他指标：\n", classification_report(dtc_y_predict, y_test, target_names=['died', 'survived']))

print("随机森林准确度:", rfc.score(x_test, y_test))
print("其他指标：\n", classification_report(rfc_y_predict, y_test, target_names=['died', 'survived']))

print("梯度提升决策树准确度:", gbc.score(x_test, y_test))
print("其他指标：\n", classification_report(gbc_y_predict, y_test, target_names=['died', 'survived']))

'''
单一决策树准确度: 0.7811550151975684
其他指标：
              precision    recall  f1-score   support

       died       0.91      0.78      0.84       236
   survived       0.58      0.80      0.67        93

avg / total       0.81      0.78      0.79       329

随机森林准确度: 0.78419452887538
其他指标：
              precision    recall  f1-score   support

       died       0.91      0.78      0.84       237
   survived       0.58      0.80      0.68        92

avg / total       0.82      0.78      0.79       329

梯度提升决策树准确度: 0.790273556231003
其他指标：
              precision    recall  f1-score   support

       died       0.92      0.78      0.84       239
   survived       0.58      0.82      0.68        90

avg / total       0.83      0.79      0.80       329

'''
