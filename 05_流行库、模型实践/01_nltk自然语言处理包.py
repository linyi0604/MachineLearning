from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download()

'''

'''

sent1 = "The cat is walking in the bedroom."
sent2 = "A dog was running across the kitchen."
# 使用词袋法 将文本转化为特征向量
count_vec = CountVectorizer()
sentences = [sent1, sent2]
# 输出转化后的特征向量
# print(count_vec.fit_transform(sentences).toarray())
'''
[[0 1 1 0 1 1 0 0 2 1 0]
 [1 0 0 1 0 0 1 1 1 0 1]]
'''
# 输出转化后特征的含义
# print(count_vec.get_feature_names())
'''
['across', 'bedroom', 'cat', 'dog', 'in', 'is', 'kitchen', 'running', 'the', 'walking', 'was']
'''

# 使用nltk对文本进行语言分析
tokens1 = nltk.word_tokenize(sent1)
tokens2 = nltk.word_tokenize(sent2)
print(tokens1)
print(tokens2)
