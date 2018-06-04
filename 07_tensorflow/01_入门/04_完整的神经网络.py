import tensorflow as tf
# 利用numpy生成数据模拟数据集
from numpy.random import RandomState


# 定义一个训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 行数指定为None  会根据传入的行数动态变化
x = tf.placeholder(tf.float32, shape=(None, 2), name="x_input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y_input")

# 定义神经网络前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 利用随机数生成数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 开启会话
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    # 初始化变量
    sess.run(init_op)
    print("训练之前的权重w1和2:")
    print(sess.run(w1))
    print(sess.run(w2))

    # 设定迭代次数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选batch个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        # 选取样本训练神经网络
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        # 每隔一段时间计算所有数据上的交叉熵并输出
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("第%d次迭代:交叉熵为%s" % (i, total_cross_entropy))
    print("训练之后的权重w1和w2分别是:")
    print(sess.run(w1))
    print(sess.run(w2))


'''
练之前的权重w1和2:
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
 
第0次迭代:交叉熵为0.067492485
第1000次迭代:交叉熵为0.016338505
第2000次迭代:交叉熵为0.009075474
第3000次迭代:交叉熵为0.007144361
第4000次迭代:交叉熵为0.005784708

训练之后的权重w1和w2分别是:
[[-1.9618274  2.582354   1.6820377]
 [-3.4681718  1.0698233  2.11789  ]]
[[-1.8247149]
 [ 2.6854665]
 [ 1.418195 ]]
'''