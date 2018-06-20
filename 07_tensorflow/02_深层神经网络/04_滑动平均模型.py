import tensorflow as tf

'''
滑动平均模型使用
'''

# 定义一个变量用于计算滑动平均 初始化为0
v1 = tf.Variable(0, dtype=tf.float32)
# 这里step模拟神经网络迭代论书,可以用于控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均类 初始化衰减率0.99 和 控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个滑动平均的操作,给定一个列表,每次执行操作,会更新列表
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 初始化时候 v1和v1的滑动平均都是0
    print(sess.run([v1, ema.average(v1)]))  # [0.0, 0.0]

    # 更新变量v1的数值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值
    # 衰减率为: min{ 0.99, (1+step)/(10+step) = 0.1 } = 0.1
    # 所以v1的滑动平均会更新为 0.1 * 0 + 0.9 * 5 = 4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # [5.0, 4.5]
    sess.run(tf.assign(v1, 10))
    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1为10

    # 更新v1的滑动平均
    # 衰减率为 min{ 0.99, (1+step)/(10+step) = 0.999 } = 0.99
    # 滑动平均更新为 0.99*4.5 + 0.01*10 = 4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))  # [10.0, 4.555]

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)])) # [10.0, 4.60945]

