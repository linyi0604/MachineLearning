# 导入手写字体加载器
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

'''
支持向量机
根据训练样本的分布，搜索所有可能的线性分类器最佳的一个。
从高纬度的数据中筛选最有效的少量训练样本。
节省数据内存，提高预测性能
但是付出更多的cpu和计算时间
'''

'''
1 获取数据
'''
# 通过数据加载器获得手写字体数字的数码图像数据并存储在digits变量中
digits = load_digits()
# 查看数据的特征维度和规模
# print(digits.data.shape)  # (1797, 64)

'''
2 分割训练集合和测试集合
'''
x_train, x_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.25,
                                                    random_state=33)

'''
3 使用支持向量机分类模型对数字图像进行识别
'''
# 对训练数据和测试数据进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# 初始化线性假设的支持向量机分类器
lsvc = LinearSVC()
# 进行训练
lsvc.fit(x_train, y_train)
# 利用训练好的模型对测试集合进行预测 测试结果存储在y_predict中
y_predict = lsvc.predict(x_test)

'''
4 支持向量机分类器 模型能力评估
'''
print("准确率：", lsvc.score(x_test, y_test))
print("其他评估数据：\n", classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))
'''
准确率： 0.9488888888888889
其他评估数据：  精确率      召回率  f1指标     数据个数
              precision    recall  f1-score   support

          0       0.92      0.97      0.94        35
          1       0.95      0.98      0.96        54
          2       0.98      1.00      0.99        44
          3       0.93      0.93      0.93        46
          4       0.97      1.00      0.99        35
          5       0.94      0.94      0.94        48
          6       0.96      0.98      0.97        51
          7       0.90      1.00      0.95        35
          8       0.98      0.83      0.90        58
          9       0.95      0.91      0.93        44

avg / total       0.95      0.95      0.95       450
'''