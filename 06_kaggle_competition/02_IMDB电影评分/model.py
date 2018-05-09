import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# import nltk
# nltk.download("stopwords")


# 从本地读取文件
train = pd.read_csv("./data/labeledTrainData.tsv", delimiter="\t")
test = pd.read_csv("./data/testData.tsv", delimiter="\t")

# 检查一下前几条数据
# print(train.head())
# print(test.head())

# 定义review_to_text函数 完成对原始评论的三项数据预处理任务
def review_to_text(review, remove_stopwords):
    # 1 去掉html标记
    raw_text = BeautifulSoup(review, "html").get_text()
    # 2 去掉非字母字符
    letters = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters.lower().split()
    # 如果remove_stopwords被激活 会去掉评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words]

    return words

# 分别对训练数据和测试数据进行内容的处理
x_train, x_test = [], []
for review in train["review"]:
    x_train.append(" ".join(review_to_text(review, True)))


for review in test["review"]:
    x_test.append(" ".join(review_to_text(review, True)))

y_train = train["sentiment"]


# 利用pipeline搭建两个朴素贝耶斯分类期,分别使用CountVectorizer 和 TfidVectorizer
pip_count = Pipeline([
    ("count_vec", CountVectorizer(analyzer="word")),
    ("mnb", MultinomialNB())
])
params_count = {
    "count_vec__binary": [True, False],
    "count_vec__ngram_range": [(1, 1), (1, 2)],
    "mnb__alpha": [0.1, 1.0, 10.1]
}

pip_tfid = Pipeline([
    ("tfid_vec", TfidfVectorizer(analyzer="word")),
    ("mnb", MultinomialNB())
])
params_tfid = {
    "tfid_vec__binary": [True, False],
    "tfid_vec__ngram_range": [(1, 1), (1, 2)],
    "mnb__alpha": [0.1, 1.0, 10.1]
}

# 使用4折交叉验证对CountVectorizer的贝耶斯分类期进行超参数搜索
gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=5, verbose=1)
gs_count.fit(x_train, y_train)  # 学习
# 输出交叉验证中最高的得分和超参数组合
print(gs_count.best_score_)
print(gs_count.best_params_)









