from sklearn.feature_extraction.text import CountVectorizer
import nltk
# nltk.download("punkt")
# nltk.download('averaged_perceptron_tagger')

# 博客: http://www.cnblogs.com/Lin-Yi/p/9006813.html

'''
分别使用词袋法和nltk自然预言处理包提供的文本特征提取
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
# 对句子词汇分割和正则化 把aren't 分割成 are 和 n't   I'm 分割成 I和'm
tokens1 = nltk.word_tokenize(sent1)
tokens2 = nltk.word_tokenize(sent2)
# print(tokens1)
# print(tokens2)
'''
['The', 'cat', 'is', 'walking', 'in', 'the', 'bedroom', '.']
['A', 'dog', 'was', 'running', 'across', 'the', 'kitchen', '.']
'''
# 整理词汇表 按照ASCII的顺序排序
vocab_1 = sorted(set(tokens1))
vocab_2 = sorted(set(tokens2))
# print(vocab_1)
# print(vocab_2)
'''
['.', 'The', 'bedroom', 'cat', 'in', 'is', 'the', 'walking']
['.', 'A', 'across', 'dog', 'kitchen', 'running', 'the', 'was']
'''
# 初始化stemer 寻找每个单词最原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens1]
stem_2 = [stemmer.stem(t) for t in tokens2]
# print(stem_1)
# print(stem_2)
'''
['the', 'cat', 'is', 'walk', 'in', 'the', 'bedroom', '.']
['A', 'dog', 'wa', 'run', 'across', 'the', 'kitchen', '.']
'''
# 利用词性标注器 对词性进行标注
pos_tag_1 = nltk.tag.pos_tag(tokens1)
pos_tag_2 = nltk.tag.pos_tag(tokens2)
# print(pos_tag_1)
# print(pos_tag_2)
'''
[('The', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('walking', 'VBG'), ('in', 'IN'), ('the', 'DT'), ('bedroom', 'NN'), ('.', '.')]
[('A', 'DT'), ('dog', 'NN'), ('was', 'VBD'), ('running', 'VBG'), ('across', 'IN'), ('the', 'DT'), ('kitchen', 'NN'), ('.', '.')]
'''