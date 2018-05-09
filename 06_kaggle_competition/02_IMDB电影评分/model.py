import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

import nltk
nltk.download("stopwords")


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
x_train = []
for review in train["review"]:
    x_train.append(" ".join(review_to_text(review, True)))

y_train = train["sentiment"]

print(x_train, y_train)