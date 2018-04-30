from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
# 导入文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 模型评估模块
from sklearn.metrics import classification_report

'''
博文：http://www.cnblogs.com/Lin-Yi/p/8970522.htmlfda

朴素贝叶斯模型广泛用于海量互联网文本分类任务。
由于假设特征条件相互独立，预测需要估计的参数规模从幂指数量级下降接近线性量级，节约内存和计算时间
但是 该模型无法将特征之间的联系考虑，数据关联较强的分类任务表现不好。
'''

'''
1 读取数据部分
'''
# 该api会即使联网下载数据
news = fetch_20newsgroups(subset="all")
# 检查数据规模和细节
# print(len(news.data))
# print(news.data[0])
'''
18846

From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>
Subject: Pens fans reactions
Organization: Post Office, Carnegie Mellon, Pittsburgh, PA
Lines: 12
NNTP-Posting-Host: po4.andrew.cmu.edu

I am sure some bashers of Pens fans are pretty confused about the lack
of any kind of posts about the recent Pens massacre of the Devils. Actually,
I am  bit puzzled too and a bit relieved. However, I am going to put an end
to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
are killing those Devils worse than I thought. Jagr just showed you why
he is much better than his regular season stats. He is also a lot
fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
regular season game.          PENS RULE!!!
'''

'''
2 分割数据部分
'''
x_train, x_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=0.25,
                                                    random_state=33)

'''
3 贝叶斯分类器对新闻进行预测
'''
# 进行文本转化为特征
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
# 初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 训练集合上进行训练， 估计参数
mnb.fit(x_train, y_train)
# 对测试集合进行预测 保存预测结果
y_predict = mnb.predict(x_test)

'''
4 模型评估
'''
print("准确率:", mnb.score(x_test, y_test))
print("其他指标：\n",classification_report(y_test, y_predict, target_names=news.target_names))
'''
准确率: 0.8397707979626485
其他指标：
                           precision    recall  f1-score   support

             alt.atheism       0.86      0.86      0.86       201
           comp.graphics       0.59      0.86      0.70       250
 comp.os.ms-windows.misc       0.89      0.10      0.17       248
comp.sys.ibm.pc.hardware       0.60      0.88      0.72       240
   comp.sys.mac.hardware       0.93      0.78      0.85       242
          comp.windows.x       0.82      0.84      0.83       263
            misc.forsale       0.91      0.70      0.79       257
               rec.autos       0.89      0.89      0.89       238
         rec.motorcycles       0.98      0.92      0.95       276
      rec.sport.baseball       0.98      0.91      0.95       251
        rec.sport.hockey       0.93      0.99      0.96       233
               sci.crypt       0.86      0.98      0.91       238
         sci.electronics       0.85      0.88      0.86       249
                 sci.med       0.92      0.94      0.93       245
               sci.space       0.89      0.96      0.92       221
  soc.religion.christian       0.78      0.96      0.86       232
      talk.politics.guns       0.88      0.96      0.92       251
   talk.politics.mideast       0.90      0.98      0.94       231
      talk.politics.misc       0.79      0.89      0.84       188
      talk.religion.misc       0.93      0.44      0.60       158

             avg / total       0.86      0.84      0.82      4712
'''