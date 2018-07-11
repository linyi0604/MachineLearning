# coding:utf8

'''
softmax回归：
    从logistic回归模型而来
    是一个多分类器
    得到属于每一个类别的概率

y = softmax(wx+b)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# 创建占位符表示待识别的图片特征和真实结果
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义softmax参数 系数w 和 偏置b
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 输出结果
y = tf.nn.softmax(tf.matmul(x, w) + b)


# 交叉损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 梯度下降w和b  指定学习率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 开启session
with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.initialize_all_variables())
    # 迭代1000步 进行梯度下降
    for _ in range(1000):
        # batch_xs 是100条 784列的特征数据  batch_ys 是100行10列的目标值
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 训练
        sess.run(train_step, feed_dict={
            x: batch_xs,
            y_: batch_ys
        })

    # 梯度下降结束后 训练结果
    # 正确率 argmax() 取出最大值的下标
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels
    }))
    # 0.9163


