import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from inception_resnet_v2 import inception_resnet_v2
from inception_resnet_v2 import inception_resnet_v2_arg_scope
import time

# http://download.tensorflow.org/example_images/flower_photo.tgz
# 已转换为[299, 299, 3]图像
INPUT_DATA = 'D:/SynologyDrive/books/tensorflow-google-dl-framework/chapter6/transfer/flower_processed_data.npy'
TRAIN_FILE = './save_model/inception_resnet_v2'
SAVE_PATH = './save_model/'
# https://github.com/tensorflow/models/tree/master/research/slim
TRAINED_CKPT_FILE = './inception_resnet_v2_2016_08_30.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 16
N_CLASSES = 5


# 不需要的参数
CHECKPOINT_EXCLUDE_SCOPE = 'InceptionResnetV2/AuxLogits/Logits,InceptionResnetV2/Logits/Logits'
# 需要训练的模型参数
TRAINABLE_SCOPE = CHECKPOINT_EXCLUDE_SCOPE


# 获取谷歌训练好的模型参数, 排除要重新训练的最后一层
def get_tuned_variable():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPE.split(',')]
    variables_to_restore = []
    # 枚举模型中所有参数，看是否要排除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# 获取需要训练的参数
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPE.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    # 加载处理好的数据
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_examples = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print('%d training, %d validation, %d testing' % (n_training_examples,
                                                      len(validation_labels),
                                                      len(testing_labels)))

    # 定义inception_resNet v2的输入, default_image_size = 299
    images = tf.placeholder(tf.float32,
                            [None, 299, 299, 3],
                            name='input_image')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    # 引用定义定义inception_resNet模型
    arg_scope = inception_resnet_v2_arg_scope
    with slim.arg_scope(arg_scope()):
        logits, _ = inception_resnet_v2(images, num_classes=N_CLASSES)

    # with tf.variable_scope('squeeze_logits'):
    #     logits = tf.squeeze(logits, axis=[1, 2])

    trainable_var = get_trainable_variables()

    # 损失函数
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits)
    # 训练
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())
    # 只训练最后一层
    # train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss(),
    #                                                               var_list=get_trainable_variables())

    # 正确率
    with tf.variable_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        # 加载之前训练的参数继续训练
        variables_to_restore = slim.get_model_variables()
        print('continue training from %s' % ckpt)
        step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        step = int(step)
        ckpt = ckpt.model_checkpoint_path
    else:
        # 没有训练数据，就先迁移一部分训练好的
        ckpt = TRAINED_CKPT_FILE
        variables_to_restore = get_tuned_variable()
        print('loading tuned variables from %s' % TRAINED_CKPT_FILE)
        step = 0

    load_fn = slim.assign_from_checkpoint_fn(ckpt, variables_to_restore, ignore_missing_vars=True)

    # 开启会话训练
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化所有参数
        init = tf.global_variables_initializer()
        sess.run(init)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(step + 1, step + 1 + STEPS):
            start_time = time.time()
            # 运行训练,不会更新所有参数
            sess.run(train_step, feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]})
            duration = time.time() - start_time

            print('current train step duration %.3f' % duration)

            # 输出日志
            if i % 100 == 0:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    images: validation_images, labels: validation_labels
                })
                print('Step %d Validation accuracy = %.1f%%' % (i, validation_accuracy * 100.0))

            start = end
            if start == n_training_examples:
                start = 0

            end = start + BATCH
            if end > n_training_examples:
                end = n_training_examples

        # 在测试集上测试正确率
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100.0))


if __name__ == '__main__':
    tf.app.run()
