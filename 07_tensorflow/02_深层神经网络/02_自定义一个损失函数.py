import tensorflow as tf
from numpy.random import RandomState

'''
模拟一个回归案例
自定义一个损失函数为:
    当真实值y_更大的时候 loss = a(y_ - y)
    当预测值y更大的时候  loss = b(y - y_)
    
    
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(
    tf.where(
        tf.greater(y, y_),
        (y - y_) * loss_more,
        (y_ - y) * loss_less
    ))
    
tf.reduce_sum()    求平均数
tf.where(condition, a, b)   condition为真时返回a 否则返回b
tf.grater(a, b)     a>b时候返回真 否则返回假
    
'''

# 一批运算的数据数量
batch_size = 8

# 输入数据有两列特征
x = tf.placeholder(tf.float32, shape=[None, 2], name="x-input")
# 输入的真实值
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y-input")

# 定义一个单层神经网络 前向传播的过程
# 权重变量 2*1维度 方差为1 均值为0 种子变量使得每次运行生成同样的随机数
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

# 计算过程
y = tf.matmul(x, w1)

# 自定义损失函数部分
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(
    tf.where(
        tf.greater(y, y_),
        (y - y_) * loss_more,
        (y_ - y) * loss_less
    ))

# 训练内容 训练速度0.001 让loss最小
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 生成随机数作为训练数据
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 预测的正确至设置为两个特征加和 加上一个噪音
# 不设置噪音 预测的意义就不大了
# 噪音设置为均值为0的极小量
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# 开启会话训练
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(
            train_step,
            feed_dict={
                x: X[start: end],
                y_: Y[start: end],
            }
        )
    print(sess.run(w1))

'''
[[1.019347 ]
 [1.0428089]]
'''