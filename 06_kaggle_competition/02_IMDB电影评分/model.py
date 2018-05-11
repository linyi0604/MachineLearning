import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import nltk.data
from gensim.models import word2vec, Word2Vec
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier



# import nltk
# nltk.download("stopwords")

'''
使用处理好的特征的数据进行建立模型
'''
# # 从本地读取文件
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

# # 分别对训练数据和测试数据进行内容的处理
# x_train, x_test = [], []
# for review in train["review"]:
#     x_train.append(" ".join(review_to_text(review, True)))
#
#
# for review in test["review"]:
#     x_test.append(" ".join(review_to_text(review, True)))
#
y_train = train["sentiment"]

#
# # 利用pipeline搭建两个朴素贝耶斯分类期,分别使用CountVectorizer 和 TfidVectorizer
# pip_count = Pipeline([
#     ("count_vec", CountVectorizer(analyzer="word")),
#     ("mnb", MultinomialNB())
# ])
# params_count = {
#     "count_vec__binary": [True, False],
#     "count_vec__ngram_range": [(1, 1), (1, 2)],
#     "mnb__alpha": [0.1, 1.0, 10.1]
# }
#
# pip_tfid = Pipeline([
#     ("tfid_vec", TfidfVectorizer(analyzer="word")),
#     ("mnb", MultinomialNB())
# ])
# params_tfid = {
#     "tfid_vec__binary": [True, False],
#     "tfid_vec__ngram_range": [(1, 1), (1, 2)],
#     "mnb__alpha": [0.1, 1.0, 10.1]
# }
#
# # 使用4折交叉验证对CountVectorizer的贝耶斯分类期进行超参数搜索
# gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
# gs_count.fit(x_train, y_train)  # 学习
# # 输出交叉验证中最高的得分和超参数组合
# print(gs_count.best_score_)
# print(gs_count.best_params_)
# '''
# Fitting 4 folds for each of 12 candidates, totalling 48 fits
# [Parallel(n_jobs=5)]: Done  48 out of  48 | elapsed:  8.0min finished
# 0.88216
# {'count_vec__ngram_range': (1, 2), 'mnb__alpha': 1.0, 'count_vec__binary': True}
# '''
# # 用最佳的模型对数据进行预测
# count_y_predict = gs_count.predict(x_test)
# # 保存结果到本地
# submission_count = pd.DataFrame({
#     "id": test["id"],
#     "sentiment": count_y_predict
# })
# submission_count.to_csv("./predict/submission_count.csv", index=False)
#
#
# # 采用4折交叉验证对TfidVectorizer朴素贝耶斯模型超参数搜索
# gs_tfid = GridSearchCV(pip_tfid, params_tfid, cv=4, n_jobs=-1, verbose=1)
# gs_tfid.fit(x_train, y_train)
# # 输出交叉验证最佳得分和超参数组合
# print(gs_tfid.best_score_)
# print(gs_tfid.best_params_)
# '''
# Fitting 4 folds for each of 12 candidates, totalling 48 fits
# [Parallel(n_jobs=-1)]: Done  48 out of  48 | elapsed:  5.3min finished
# 0.88712
# {'tfid_vec__ngram_range': (1, 2), 'tfid_vec__binary': True, 'mnb__alpha': 0.1}
# '''
# # 进行预测
# tfid_y_predict = gs_tfid.predict(x_test)
# # 保存结果到本地
# submission_tfid = pd.DataFrame({
#     "id": test["id"],
#     "sentiment": tfid_y_predict
# })
# submission_tfid.to_csv("./predict/submission_tfid.csv", index=False)



'''
使用nltk建立模型
'''
# # 从本地读取未标记的数据
# unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", delimiter="\t", quoting=3)
#
# # 准备使用nltk的tokenizer对影评中的英文句子进行分割
# tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
#
# # 定义函数review_to_sentences 对影评进行分句
# def review_to_sentences(review, tokenizer):
#     raw_sentences = tokenizer.tokenize(review.strip())
#     sentences = []
#     for raw_sentence in raw_sentences:
#         if len(raw_sentence)>0:
#             sentences.append(review_to_text(raw_sentence, False))
#     return sentences
#
# corpora = []
# # 准备用于训练词向量的数据
# for review in unlabeled_train["review"]:
#     corpora += review_to_sentences(review, tokenizer)
# # 配置训练词向量模型的超参数
num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3
# # 模型训练
# model = word2vec.Word2Vec(corpora, workers=num_workers,
#                           size=num_features,min_count=min_word_count,
#                           window=context,sample=downsampling)
# model.init_sims(replace=True)
# model_name = "./300features_20minwords_10context"
# # 可以将词向量训练结果保存在硬盘
# model.save(model_name)

# 直接从训练好的向量模型倒入训练成果
model = Word2Vec.load("./300features_20minwords_10context")

# 和man 最接近的词汇
# print(model.most_similar("man"))
'''
[('woman', 0.6180815696716309), 
('lady', 0.5649946928024292), 
('lad', 0.5551677346229553), 
('person', 0.5436856746673584), 
('guy', 0.5340112447738647), 
('millionaire', 0.5196341276168823), 
('boy', 0.5175526142120361), 
('men', 0.5128821134567261), 
('monk', 0.5069575309753418), 
('businessman', 0.5062038898468018)]
'''

# 定义一个函数使用词向量产生文本特征向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

# 定义另一个每条影评转化为基于词向量的特征向量
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

# 准备新的基于词向量表示的训练和测试特征向量
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_text(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# 使用 模型进行感情分析
gbc = GradientBoostingClassifier()
params_gbc = {
    "n_estimators":[10, 100, 500],
    "learning_rate":[0.01,0.1,1.0],
    "max_depth":[2,3,4]
}
gs_gbc = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1, verbose=1)
gs_gbc.fit(trainDataVecs, y_train)
# 输出最佳性能的超参数组合和性能得分
print(gs_gbc.best_score_)
print(gs_gbc.best_params_)

