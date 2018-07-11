# coding:utf8

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../data/mnist/")

# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
'''
# 训练集数据
(55000, 784)
(55000,)
'''

# print(mnist.validation.images.shape)
# print(mnist.validation.labels.shape)
'''
# 验证数据
(5000, 784)
(5000,)
'''

print(mnist.test.images.shape)
print(mnist.test.labels.shape)
'''
# 测试数据集
(10000, 784)
(10000,)
'''