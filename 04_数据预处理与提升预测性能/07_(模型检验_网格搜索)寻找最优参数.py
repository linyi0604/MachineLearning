from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

# 博文： http://www.cnblogs.com/Lin-Yi/p/9000989.html

'''
如何确定一个模型应该使用哪种参数？

k折交叉验证：
   将样本分成k份
   每次取其中一份做测试数据 其他做训练数据 
   一共进行k次训练和测试
   用这种方式 充分利用样本数据，评估模型在样本上的表现情况
   
   
网格搜索：
    一种暴力枚举搜索方法
    对模型参数列举出集中可能，
    对所有列举出的可能组合进行模型评估
    从而找到最好的模型参数

'''

# 联网获取所有想你问数据
news = fetch_20newsgroups(subset="all")
# 分割训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(news.data[:3000],
                                                    news.target[:3000],
                                                    test_size=0.25,
                                                    random_state=33)

# 使用pipeline简化系统搭建流程
clf = Pipeline([("vect", TfidfVectorizer(stop_words="english", analyzer="word")), ("svc", SVC())])

# 这里要实验的超参数有两个  4个svg__gama 和 3个svg__C 一共12种组合
# np.logspace(start, end, num) 从10^start 到 10^end 创建num个数的等比数列
parameters = {"svc__gamma": np.logspace(-2, 1, 4), "svc__C": np.logspace(-1, 1, 3)}

# 网格搜索
# 创建一个网格搜索: 12组参数组合， 3折交叉验证
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索
time_ = gs.fit(x_train, y_train)
print(time_)
print(gs.best_params_, gs.best_score_)
# 输出最佳模型在测试机和上的准确性
print(gs.score(x_test, y_test))
'''
Fitting 3 folds for each of 12 candidates, totalling 36 fits
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   8.3s
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    8.3s remaining:    0.0s
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   8.5s
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   8.5s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   8.4s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   8.5s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   8.5s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   8.4s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   8.6s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   8.6s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   8.5s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   8.6s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   8.7s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   8.3s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   8.4s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   8.5s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   8.3s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   8.4s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   8.5s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   8.5s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   8.6s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   8.7s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   8.5s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   8.6s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   8.7s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   8.4s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   8.4s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   8.7s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   8.6s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   8.6s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   8.6s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   8.5s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   8.6s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   9.3s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   8.8s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   8.9s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   8.7s

12组超参数 3折交叉验证 共36个搜索项 花费5.2分钟
[Parallel(n_jobs=1)]: Done  36 out of  36 | elapsed:  5.2min finished

最佳参数   最佳训练得分
{'svc__C': 10.0, 'svc__gamma': 0.1} 0.7906666666666666
最佳模型的测试得分
0.8226666666666667

'''