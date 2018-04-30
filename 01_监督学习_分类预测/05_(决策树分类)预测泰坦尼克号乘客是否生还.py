import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

'''
博文：http://www.cnblogs.com/Lin-Yi/p/8970609.html

决策树
涉及多个特征，没有明显的线性关系
推断逻辑非常直观
不需要对数据进行标准化
'''

'''
1 准备数据
'''
# 读取泰坦尼克乘客数据，已经从互联网下载到本地
titanic = pd.read_csv("../data//titanic/titanic.txt")
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
3 训练模型 进行预测
'''
# 初始化决策树分类器
dtc = DecisionTreeClassifier()
# 训练
dtc.fit(x_train, y_train)
# 预测 保存结果
y_predict = dtc.predict(x_test)

'''
4 模型评估
'''
print("准确度:", dtc.score(x_test, y_test))
print("其他指标：\n", classification_report(y_predict, y_test, target_names=['died', 'survived']))
'''
准确度: 0.7811550151975684
其他指标：
              precision    recall  f1-score   support

       died       0.91      0.78      0.84       236
   survived       0.58      0.80      0.67        93

avg / total       0.81      0.78      0.79       329
'''
