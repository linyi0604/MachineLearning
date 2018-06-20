import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST相关常数
INPUT_NODE = 784    # 输入层节点数, 等于图片的像素数
OUTPUT_NODE = 10    # 输出层节点书, 也就是类别数量

# 配置神经网络参数
LAYER1_NODE = 500   # 隐藏层节点个数, 这里只有一个隐藏层的神经网络作为样例
BATCH_SIZE = 100    # 一个训练batch中数据个数,越小越接近随即梯度下降,越大越接近梯度下降

# 随机梯度下降
LEARNING_RATE_BASE = 0.8    # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习衰减率

REGULARIZATION_RATE = 0.0001 # 正则化在损失函数中的系数
TRAINING_STEPS = 30000      # 训练轮数
MOVING_AVERAGE_DECAY = 0.99 # 滑动平均衰减率


# 一个辅助函数 用于计算当前神经网络的y前向传播的过程
# 使用reLu激活的三层全链接网络 有一个隐藏层
def inference(input_tensor, avg_class, weights1, bias1, weights2, bias2):
    # 没有滑动平均类的时候,不使用滑动平均
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + bias1)
        return tf.matmul(layer1, weights2) + bias2
    else:
        # 首先计算滑动平均,然后再计算前向传播
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(bias1))
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(bias2))


# 训练模型的过程
def train(mnist):
    # 输入的特征和真实值 利用占位符
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE], name="y-input")

    # 生成隐藏层的参数   标准差0.1
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数神经网络的前向传播 这里用于计算滑动平均的类为None
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数变量,这个变量不能被训练
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均类 给定平滑衰减和训练轮数,达到下降速度逐渐减慢的效果
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 将神经元上的所有变量使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 滑动平均不会改变原变量,会维护一个新的影子变量
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵 刻画预测值和真实值的差距
    # 多个分类有一个是正确值的时候,采用这个函数计算
    # argmax函数用于得到正确答案对应类别的编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    # 计算当前batch中所有样例的交叉熵的平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy)



# 程序的入口
def main(argv=None):
    # 此处声明MNIST数据集 如果有就读取否则下载
    mnist = input_data.read_data_sets("../../data/mnist")
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
