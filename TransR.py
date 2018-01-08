import tensorflow as tf
import copy
from data_related.data_reader import *
from data_related.batch_producer import *


class Config(object):
    def __init__(self):
        self.entity_embedding_size = 32
        self.relation_embedding_size = 32
        self.margin = 2.0
        self.relation_num = 1345
        self.entity_num = 14951
        self.batch_size = 64
        self.base_learning_rate = 0.01
        self.base_learning_rate_softmax = 1.0


class TransRModel(object):
    def __init__(self, config, is_training):
        batch_size = config.batch_size
        margin = config.margin
        entity_embedding_size = config.entity_embedding_size
        relation_embedding_size = config.relation_embedding_size
        entity_num = config.entity_num
        relation_num = config.relation_num
        base_learning_rate = config.base_learning_rate
        base_learning_rate_softmax = config.base_learning_rate_softmax

        self.pos_h = tf.placeholder(tf.int32, [batch_size])
        self.pos_r = tf.placeholder(tf.int32, [batch_size])
        self.pos_t = tf.placeholder(tf.int32, [batch_size])

        self.neg_h = tf.placeholder(tf.int32, [batch_size])
        self.neg_r = tf.placeholder(tf.int32, [batch_size])
        self.neg_t = tf.placeholder(tf.int32, [batch_size])

        with tf.device("/cpu:0"):
            self.entity_embeddings = tf.get_variable("entity_embeddings", [entity_num, entity_embedding_size], dtype=tf.float32)
            self.relation_embeddings = tf.get_variable("relation_embeddings", [relation_num, relation_embedding_size], dtype=tf.float32)
            self.relation_matrix = tf.get_variable("relation_matrix", [relation_num, entity_embedding_size * relation_embedding_size], dtype=tf.float32)

            embedded_pos_h = tf.reshape(tf.nn.embedding_lookup(self.entity_embeddings, self.pos_h), [-1, entity_embedding_size, 1])
            embedded_pos_r = tf.reshape(tf.nn.embedding_lookup(self.relation_embeddings, self.pos_r), [-1, relation_embedding_size])
            embedded_pos_t = tf.reshape(tf.nn.embedding_lookup(self.entity_embeddings, self.pos_t), [-1, entity_embedding_size, 1])

            embedded_neg_h = tf.reshape(tf.nn.embedding_lookup(self.entity_embeddings, self.neg_h), [-1, entity_embedding_size, 1])
            embedded_neg_r = tf.reshape(tf.nn.embedding_lookup(self.relation_embeddings, self.neg_r), [-1, relation_embedding_size])
            embedded_neg_t = tf.reshape(tf.nn.embedding_lookup(self.entity_embeddings, self.neg_t), [-1, entity_embedding_size, 1])

            matrix = tf.reshape(tf.nn.embedding_lookup(self.relation_matrix, self.pos_r), [-1, relation_embedding_size, entity_embedding_size])

        embedded_pos_h = tf.reshape(tf.matmul(matrix, embedded_pos_h), [-1, relation_embedding_size])
        embedded_pos_t = tf.reshape(tf.matmul(matrix, embedded_pos_t), [-1, relation_embedding_size])

        embedded_neg_h = tf.reshape(tf.matmul(matrix, embedded_neg_h), [-1, relation_embedding_size])
        embedded_neg_t = tf.reshape(tf.matmul(matrix, embedded_neg_t), [-1, relation_embedding_size])

        self.pos = tf.reduce_sum((embedded_pos_h + embedded_pos_r - embedded_pos_t) ** 2, 1, True)
        self.neg = tf.reduce_sum((embedded_neg_h + embedded_neg_r - embedded_neg_t) ** 2, 1, True)

        self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + margin, 0))

        softmax_weights = tf.get_variable('softmax_weights', [entity_embedding_size, entity_num], dtype=tf.float32)
        softmax_biases = tf.get_variable('softmax_biases', [entity_num], dtype=tf.float32)
        softmax_logits = tf.matmul(embedded_pos_h + embedded_pos_r,
                                   softmax_weights) + softmax_biases
        self.softmax_pred = tf.argmax(tf.nn.softmax(softmax_logits), axis=-1)

        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.pos_t, [batch_size]), logits=softmax_logits, name='softmax_loss'
        )
        self.softmax_loss = tf.reduce_mean(softmax_loss)

        correct_prediction = tf.equal(tf.cast(tf.argmax(softmax_logits, -1), dtype=tf.int32),
                                      tf.reshape(self.pos_t, [batch_size]))
        self.softmax_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        top10_correction_predcition = tf.nn.in_top_k(predictions=softmax_logits,
                                                     targets=tf.reshape(self.pos_t, [batch_size]), k=10)
        self.softmax_top10_accuracy = tf.reduce_mean(tf.cast(top10_correction_predcition, tf.float32))

        top100_correction_predcition = tf.nn.in_top_k(predictions=softmax_logits,
                                                      targets=tf.reshape(self.pos_t, [batch_size]), k=100)
        self.softmax_top100_accuracy = tf.reduce_mean(tf.cast(top100_correction_predcition, tf.float32))

        if is_training:
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                base_learning_rate,
                global_step,
                300,
                0.98
            )
            learning_rate_softmax = tf.train.exponential_decay(
                base_learning_rate_softmax,
                global_step,
                300,
                0.98
            )

            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            self.softmax_train_op = tf.train.GradientDescentOptimizer(learning_rate_softmax).minimize(self.softmax_loss)


def main():
    config = Config()
    entity2id_file = "dataset/FB15K/entity2id.txt"
    relation2id_file = "dataset/FB15K/relation2id.txt"
    train2id_file = "dataset/FB15K/triple2id.txt"
    test2id_file = "dataset/FB15K/test2id.txt"
    valid2id_file = "dataset/FB15K/valid2id.txt"

    entity2id_dic = read_entity2id(entity2id_file)
    relation2id_dic = read_relation2id(relation2id_file)
    train_triple = read_triple(train2id_file)
    test_triple = read_triple(test2id_file)
    valid_triple = read_triple(valid2id_file)

    train_record_file = "data/train.tfrecords"
    test_record_file = "data/test.tfrecords"
    # tfrecord_write(train_triple, train_record_file)
    # tfrecord_write(test_triple, test_record_file)
    train_head_raw, train_relation_raw, train_tail_raw = tfrecord_read(train_record_file)
    test_head_raw, test_relation_raw, test_tail_raw = tfrecord_read(test_record_file)
    train_head_batch, train_relation_batch, train_tail_batch = tf.train.shuffle_batch([train_head_raw, train_relation_raw, train_tail_raw],
                                                                                      batch_size=config.batch_size,
                                                                                      capacity=483142,
                                                                                      min_after_dequeue=1000)
    train_head_batch=tf.reshape(train_head_batch,[-1])

    test_head_batch, test_relation_batch, test_tail_batch = tf.train.shuffle_batch([test_head_raw, test_relation_raw, test_tail_raw],
                                                                                   batch_size=config.batch_size,
                                                                                   capacity=59071,
                                                                                   min_after_dequeue=1000)

    with tf.name_scope('Train'):
        with tf.variable_scope("Model", reuse=None):
            model = TransRModel(config, is_training=True)

    with tf.name_scope('Test'):
        with tf.variable_scope("Model", reuse=True):
            test_model = TransRModel(config, is_training=False)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10000000):
            # training_input = sess.run(training_input_batch)
            train_head, train_relation, train_tail = sess.run([train_head_batch, train_relation_batch, train_tail_batch])

            neg_tail = copy.copy(train_tail)
            for j in range(len(train_tail)):
                neg_tail[j] += 1
                neg_tail[j] %= config.entity_num

            loss, opt = sess.run([model.loss, model.train_op], feed_dict={
                model.pos_h: train_head,
                model.pos_r: train_relation,
                model.pos_t: train_tail,
                model.neg_h: train_head,
                model.neg_r: train_relation,
                model.neg_t: neg_tail
            })

            softmax_loss, softmax_opt = sess.run([model.softmax_loss, model.softmax_train_op], feed_dict={
                model.pos_h: train_head,
                model.pos_r: train_relation,
                model.pos_t: train_tail,
                model.neg_h: train_head,
                model.neg_r: train_relation,
                model.neg_t: neg_tail
            })


            if i % 1000 == 0:
                test_head, test_relation, test_tail = sess.run([test_head_batch, test_relation_batch, test_tail_batch])
                neg_test_tail = copy.copy(test_tail)
                for k in range(len(train_tail)):
                    neg_test_tail[k] += 1
                    neg_test_tail %= config.entity_num

                test_loss, test_softmax_loss = sess.run([test_model.loss, test_model.softmax_loss], feed_dict={
                    test_model.pos_h: test_head,
                    test_model.pos_r: test_relation,
                    test_model.pos_t: test_tail,
                    test_model.neg_h: test_head,
                    test_model.neg_r: test_relation,
                    test_model.neg_t: neg_test_tail
                })

                top_1_acc_train, top_10_acc_train, top_100_acc_train = sess.run(
                    [model.softmax_accuracy, model.softmax_top10_accuracy,
                     model.softmax_top100_accuracy], feed_dict={
                        model.pos_h: train_head,
                        model.pos_r: train_relation,
                        model.pos_t: train_tail,
                        model.neg_h: train_head,
                        model.neg_r: train_relation,
                        model.neg_t: neg_tail
                    })

                top_1_acc, top_10_acc, top_100_acc = sess.run(
                    [test_model.softmax_accuracy, test_model.softmax_top10_accuracy,
                     test_model.softmax_top100_accuracy], feed_dict={
                        test_model.pos_h: test_head,
                        test_model.pos_r: test_relation,
                        test_model.pos_t: test_tail,
                        test_model.neg_h: test_head,
                        test_model.neg_r: test_relation,
                        test_model.neg_t: neg_test_tail
                    })

                print("************************************************")
                print("step: " + str(i))

                print("train_loss: ")
                print(loss)
                print("train_softmax_loss: ")
                print(softmax_loss)
                print("test_loss: ")
                print(test_loss)
                print("test_softmax_loss: ")
                print(test_softmax_loss)

                print("-----------------------------------------------")

                print("train_top_1_acc: ")
                print(top_1_acc_train)
                print("train_top_10_acc: ")
                print(top_10_acc_train)
                print("train_top_100_acc: ")
                print(top_100_acc_train)
                print("-----------------------------------------------")

                print("top_1_acc: ")
                print(top_1_acc)
                print("top_10_acc: ")
                print(top_10_acc)
                print("top_100_acc: ")
                print(top_100_acc)
                print("************************************************")

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()





