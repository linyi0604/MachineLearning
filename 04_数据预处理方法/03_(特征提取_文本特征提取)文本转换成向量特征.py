from sklearn.datasets import  fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 博文：http://www.cnblogs.com/Lin-Yi/p/8974108.html

'''
文本特征提取：
    将文本数据转化成特征向量的过程
    比较常用的文本特征表示法为词袋法
词袋法：
    不考虑词语出现的顺序，每个出现过的词汇单独作为一列特征
    这些不重复的特征词汇集合为词表
    每一个文本都可以在很长的词表上统计出一个很多列的特征向量
    如果每个文本都出现的词汇，一般被标记为 停用词 不计入特征向量
    
主要有两个api来实现 CountVectorizer 和 TfidfVectorizer
CountVectorizer：
    只考虑词汇在文本中出现的频率
TfidfVectorizer：
    除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量
    能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征

相比之下，文本条目越多，Tfid的效果会越显著


下面对两种提取特征的方法，分别设置停用词和不停用，
使用朴素贝叶斯进行分类预测，比较评估效果

'''


# 1 下载新闻数据
news = fetch_20newsgroups(subset="all")


# 2 分割训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.25,
                                                    random_state=33)


# 3.1 采用普通统计CountVectorizer提取特征向量
# 默认配置不去除停用词
count_vec = CountVectorizer()
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)
# 去除停用词
count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
x_count_stop_train = count_stop_vec.fit_transform(x_train)
x_count_stop_test = count_stop_vec.transform(x_test)

# 3.2 采用TfidfVectorizer提取文本特征向量
# 默认配置不去除停用词
tfid_vec = TfidfVectorizer()
x_tfid_train = tfid_vec.fit_transform(x_train)
x_tfid_test = tfid_vec.transform(x_test)
# 去除停用词
tfid_stop_vec = TfidfVectorizer(analyzer='word', stop_words='english')
x_tfid_stop_train = tfid_stop_vec.fit_transform(x_train)
x_tfid_stop_test = tfid_stop_vec.transform(x_test)


# 4 使用朴素贝叶斯分类器  分别对两种提取出来的特征值进行学习和预测
# 对普通通统计CountVectorizer提取特征向量 学习和预测
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)   # 学习
mnb_count_y_predict = mnb_count.predict(x_count_test)   # 预测
# 去除停用词
mnb_count_stop = MultinomialNB()
mnb_count_stop.fit(x_count_stop_train, y_train)   # 学习
mnb_count_stop_y_predict = mnb_count_stop.predict(x_count_stop_test)    # 预测

# 对TfidfVectorizer提取文本特征向量 学习和预测
mnb_tfid = MultinomialNB()
mnb_tfid.fit(x_tfid_train, y_train)
mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
# 去除停用词
mnb_tfid_stop = MultinomialNB()
mnb_tfid_stop.fit(x_tfid_stop_train, y_train)   # 学习
mnb_tfid_stop_y_predict = mnb_tfid_stop.predict(x_tfid_stop_test)    # 预测

# 5 模型评估
# 对普通统计CountVectorizer提取的特征学习模型进行评估
print("未去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count.score(x_count_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_count_y_predict, y_test))
print("去除停用词的CountVectorizer提取的特征学习模型准确率：", mnb_count_stop.score(x_count_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_count_stop_y_predict, y_test))

# 对TfidVectorizer提取的特征学习模型进行评估
print("TfidVectorizer提取的特征学习模型准确率：", mnb_tfid.score(x_tfid_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_tfid_y_predict, y_test))
print("去除停用词的TfidVectorizer提取的特征学习模型准确率：", mnb_tfid_stop.score(x_tfid_stop_test, y_test))
print("更加详细的评估指标:\n", classification_report(mnb_tfid_stop_y_predict, y_test))

'''
未去除停用词的CountVectorizer提取的特征学习模型准确率： 0.8397707979626485
更加详细的评估指标:
              precision    recall  f1-score   support

          0       0.86      0.86      0.86       201
          1       0.86      0.59      0.70       365
          2       0.10      0.89      0.17        27
          3       0.88      0.60      0.72       350
          4       0.78      0.93      0.85       204
          5       0.84      0.82      0.83       271
          6       0.70      0.91      0.79       197
          7       0.89      0.89      0.89       239
          8       0.92      0.98      0.95       257
          9       0.91      0.98      0.95       233
         10       0.99      0.93      0.96       248
         11       0.98      0.86      0.91       272
         12       0.88      0.85      0.86       259
         13       0.94      0.92      0.93       252
         14       0.96      0.89      0.92       239
         15       0.96      0.78      0.86       285
         16       0.96      0.88      0.92       272
         17       0.98      0.90      0.94       252
         18       0.89      0.79      0.84       214
         19       0.44      0.93      0.60        75

avg / total       0.89      0.84      0.86      4712

去除停用词的CountVectorizer提取的特征学习模型准确率： 0.8637521222410866
更加详细的评估指标:
              precision    recall  f1-score   support

          0       0.89      0.85      0.87       210
          1       0.88      0.62      0.73       352
          2       0.22      0.93      0.36        59
          3       0.88      0.62      0.73       341
          4       0.85      0.93      0.89       222
          5       0.85      0.82      0.84       273
          6       0.79      0.90      0.84       226
          7       0.91      0.91      0.91       239
          8       0.94      0.98      0.96       264
          9       0.92      0.98      0.95       236
         10       0.99      0.92      0.95       251
         11       0.97      0.91      0.93       254
         12       0.89      0.87      0.88       254
         13       0.95      0.94      0.95       248
         14       0.96      0.91      0.93       233
         15       0.94      0.87      0.90       250
         16       0.96      0.89      0.93       271
         17       0.98      0.95      0.97       238
         18       0.90      0.84      0.87       200
         19       0.53      0.91      0.67        91

avg / total       0.90      0.86      0.87      4712

TfidVectorizer提取的特征学习模型准确率： 0.8463497453310697
更加详细的评估指标:
              precision    recall  f1-score   support

          0       0.67      0.84      0.75       160
          1       0.74      0.85      0.79       218
          2       0.85      0.82      0.83       256
          3       0.88      0.76      0.82       275
          4       0.84      0.94      0.89       217
          5       0.84      0.96      0.89       229
          6       0.69      0.93      0.79       192
          7       0.92      0.84      0.88       259
          8       0.92      0.98      0.95       259
          9       0.91      0.96      0.94       238
         10       0.99      0.88      0.93       264
         11       0.98      0.73      0.83       321
         12       0.83      0.91      0.87       226
         13       0.92      0.97      0.95       231
         14       0.96      0.89      0.93       239
         15       0.97      0.51      0.67       443
         16       0.96      0.83      0.89       293
         17       0.97      0.92      0.95       245
         18       0.62      0.98      0.76       119
         19       0.16      0.93      0.28        28

avg / total       0.88      0.85      0.85      4712

去除停用词的TfidVectorizer提取的特征学习模型准确率： 0.8826400679117148
更加详细的评估指标:
              precision    recall  f1-score   support

          0       0.81      0.86      0.83       190
          1       0.81      0.85      0.83       238
          2       0.87      0.84      0.86       257
          3       0.88      0.78      0.83       269
          4       0.90      0.92      0.91       235
          5       0.88      0.95      0.91       243
          6       0.80      0.90      0.85       230
          7       0.92      0.89      0.90       244
          8       0.94      0.98      0.96       265
          9       0.93      0.97      0.95       242
         10       0.99      0.88      0.93       264
         11       0.98      0.85      0.91       273
         12       0.86      0.93      0.89       231
         13       0.93      0.96      0.95       237
         14       0.97      0.90      0.93       239
         15       0.96      0.70      0.81       320
         16       0.98      0.84      0.90       294
         17       0.99      0.92      0.95       248
         18       0.74      0.97      0.84       145
         19       0.29      0.96      0.45        48

avg / total       0.90      0.88      0.89      4712
'''