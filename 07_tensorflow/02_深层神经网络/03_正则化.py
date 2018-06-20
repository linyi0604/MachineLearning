import tensorflow as tf

'''
一个带有L2正则化的损失函数
计算5层神经网络
'''

# 输入特征x  2列特征 若干行
x = tf.placeholder(tf.float32, shape=(None, 2))
# 输入预测结果真实值 1列结果 若干行
y_ = tf.placeholder(tf.float32, shape=(None, 1))
# 每批处理8个数据
batch_size = 8

# 定义每层网络中节点个数
layer_dimension = [2, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护前向传播时最深层的节点,开始是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 获取神经网络上一层的权重,并将这个权重l2正则化损失加入losses集合
def get_weight(shape, lbd):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    # 加入到losess集合
    tf.add_to_collection(
        "losses",
        tf.contrib.layers.l2_regularizer(lbd)(var)
    )
    return var

# 通过一个循环,建立一个5层全链接的神经网络
for i in range(1, n_layers):
    # 下一层节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层的权重变量,并使用l2正则化损失加入计算图的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用reLu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层,更新当前节点个数
    in_dimension = layer_dimension[i]


# 正则化已经加入了losses集合, 这里只需计算刻画模型上损失
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差加入到集合
tf.add_to_collection("losses", mse_loss)

loss = tf.add_n(tf.get_collection("losses"))
