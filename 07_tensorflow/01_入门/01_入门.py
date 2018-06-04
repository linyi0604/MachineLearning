import tensorflow as tf


'''
计算图

tensorflow是一个通过计算图的形式来表示计算的编程系统
tensorflow中每一个计算都是计算图上的一个节点
节点之间的边描述了计算之间的依赖关系
'''
# 1 计算图的使用
# 获得系统默认的计算图
# print(tf.get_default_graph)     # <function get_default_graph at 0x7f484912be18>
# 生成新的计算图
g1 = tf.Graph()
with g1.as_default():
    # 在计算图g1中定义变量v 设置为0
    v = tf.get_variable(name="v",
                        initializer=tf.zeros_initializer(),
                        shape=[1])
g2 = tf.Graph()
with g2.as_default():
    # 在图g2中定义变量v 初始值1
    v = tf.get_variable(name="v",
                        initializer=tf.ones_initializer(),
                        shape=[1])

# 在图1中读取变量v的值
with tf.Session(graph=g1) as sess:
    # 初始化所有变量
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable(name="v")))   # [0.]

# 在图2中读取变量v的值
with tf.Session(graph=g2) as sess:
    # 初始化所有变量
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable(name="v")))  # [1.]


g = tf.Graph()
# 指定运行设备
with g.device("/gpu:0"):
    pass



'''
张量 tensor
张量可以简单理解成多维数组
零阶张量为 标量 scala   也就是一个数
n阶张量可以理解为n维数组

张量没有保存真正的数字 而是保存一个结果运算过程的引用 并不会执行加法运算
获得一个张量 使用tf.constant(value, name, shape, dtype)
dtype为数值类型,不同类型之间不能进行操作

'''
# tf.constant 是一个计算 结果为一个张量
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")
# print(result)   # Tensor("add:0", shape=(2,), dtype=float32)


'''
会话 session
会话用来执行定义好的运算
管理tensorflow运行时的资源
计算完成后资源回收
会话指定之后 可以通过tf.Tensor.eval() 来计算一个张量的取值
'''
# 开启会话
with tf.Session() as sess:
    with sess.as_default(): # 注册默认会话 计算张量的值
        # result为之前张量a和b的加法引用
        print(result.eval())    # [3. 5.]


'''
变量
tf.Variable
作用是保存和更新神经网络中的参数
经常用随机数来初始化变量
常用的随机数生成器:
    tf.random_normal 正太分布
    tf.truncated_normal 正太分布 如果平均值超过两个标准差就重新随机
    tf.random_uniform 均匀分布
    tf.random_gamma Gamma分布
    tf.zeros 产生全0数组
    tf.ones 产生全1数组
    tf.fill 给定数字数组
    tf.constant 定量值
'''
# 声明一个变量 随机生成 2*3的矩阵 满足正太分布 均值为0 标准差为2
weights = tf.Variable(tf.random_normal(shape=[2,3], stddev=2, mean=0))
# 生成三个零的数组变量
bias = tf.Variable(tf.zeros(shape=[3]))
# 也支持用其他变量初始化的形式声明变量
# 与weight相同
w2 = tf.Variable(weights.initialized_value())
# 是weight的两倍
w3 = tf.Variable(weights.initialized_value()*2)


