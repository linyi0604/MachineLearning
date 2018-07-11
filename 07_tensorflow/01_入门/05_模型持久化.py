import tensorflow as tf

# 声明两个变量计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1], name="v1"))
v2 = tf.Variable(tf.constant(2.0, shape=[1], name="v2"))
result = v1 + v2

init_op = tf.initialize_all_variables()

# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # 将模型保存
    saver.save(sess, "./model/model.ckpt")


# 加载保存好的模型代码和保存模型的代码要保持一致
with tf.Session() as sess:
    # 加载保存的模型继续执行
    saver.restore(sess, "./model/model.ckpt")
    print(sess.run(result)) # [3.]

# 直接加载持久化的图,不需要与保存之前的代码保持一致
saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess, "./model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))) # [3.]