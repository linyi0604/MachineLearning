import tensorflow as tf


'''
模拟一个前向传播过程
'''
# 声明w1 w2两个变量,通过seed参数设定随机种子 保证每次运行得到一样的结果
# 2*3 的矩阵 标准差为1 均值默认为0
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入特征向量定义为一个常量 1*2 矩阵
x = tf.constant([[0.7, 0.9]])
# 获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 开启会话进行运算
with tf.Session() as sess:
    # 对w1和w2进行初始化
    # sess.run(w1.initializer)
    # sess.run(w2.initializer)
    # 也可以初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 计算结果
    print(sess.run(y))  # [[3.957578]]
