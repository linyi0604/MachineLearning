from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import nltk, re
from gensim.models import word2vec

# nltk.download('punkt')


'''
词向量技术 Word2Vec 
    每个连续词汇片段都会对后面有一定制约 称为上下文context
    
    找到句子之间语义层面的联系
    
'''

# 联网下载新闻数据
news = fetch_20newsgroups(subset="all")
x, y = news.data, news.target

# 定义一个函数 将每条新闻中的句子分离,并返回一个句子的列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        temp = re.sub("[^a-zA-Z]", " ", sent.lower().strip()).split()
        sentences.append(temp)

    return sentences

# 将长新闻中的句子剥离出来用于训练
sentences = []
for i in x:
    sentence_list = news_to_sentences(i)
    sentences += sentence_list


# 配置词向量的维度
num_features = 300
# 保证被考虑的词汇的频度
min_word_count = 20
# 并行计算使用cpu核心数量
num_workers = 2
# 定义训练词向量的上下文窗口大小
context = 5
downsapling = 1e-3

# 训练词向量模型
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsapling)
# 这个设定代表当前训练好的词向量为最终版, 也可以加速模型训练的速度
model.init_sims(replace=True)

# 利用训练好的模型 寻找文本中与college相关的十个词汇
print(model.most_similar("college"))
'''
[('wisconsin', 0.7664438486099243), 
('osteopathic', 0.7474539279937744), 
('madison', 0.7433826923370361), 
('univ', 0.7296794652938843), 
('melbourne', 0.7212647199630737), 
('walla', 0.7068545818328857), 
('maryland', 0.7038443088531494), 
('carnegie', 0.7038302421569824), 
('institute', 0.7003713846206665), 
('informatics', 0.6968873143196106)]
'''