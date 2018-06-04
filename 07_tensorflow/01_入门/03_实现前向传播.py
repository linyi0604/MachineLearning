import tensorflow as tf

# 2*3 矩阵变量 初始化为 随机标准差为1
w1 = tf.Variable(tf.random_normal(shape=[2, 3], stddev=1))
# 3*1 矩阵变量 初始化为 随机标准差为1
w2 = tf.Variable(tf.random_normal(shape=[3, 1], stddev=1))

# 定义placeholder 存放输入数据的地方
x = tf.placeholder(tf.float32, shape=[1, 2], name="input")

# 定义计算过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 初始化所有变量
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    # x是个占位符, feed_dict 为要给x读入的数据
    print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
