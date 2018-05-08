import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 博客: http://www.cnblogs.com/Lin-Yi/p/9009271.html


'''
XGBoost
提升分类器
    属于集成学习模型
    把成百上千个分类准确率较低的树模型组合起来
    不断迭代,每次迭代生成一颗新的树
    
    
下面 对泰坦尼克遇难预测
使用XGBoost模型 和 其他分类器性能进行比较

'''

titanic = pd.read_csv("../data/titanic/titanic.txt")
# 抽取pclass age 和 sex 作为训练样本
x = titanic[["pclass", "age", "sex"]]
y = titanic["survived"]
# 采集的age空的用平均数补全
x["age"].fillna(x["age"].mean(), inplace=True)

# 分割训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=33)
# 提取字典特征 进行 向量化
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient="record"))
x_test = vec.transform(x_test.to_dict(orient="record"))

# 采用默认配置的随机森林进行预测
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
print("随机森林预测准确率:", rfc.score(x_test, y_test))  # 0.7811550151975684

# 采用XGBoost模型进行预测
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)
print("XGBoost预测准确率:", xgbc.score(x_test, y_test))  # 0.7872340425531915



