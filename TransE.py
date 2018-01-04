import tensorflow as tf
from data_related.data_reader import *


class Config(object):
    def __init__(self):
        self.test_flag = False
        self.entity_embedding_size = 128
        self.relation_embedding_size = 128
        self.margin = 1.0
        self.relation_num = 0
        self.entity_num = 0
        self.batch_size = 32
        self.base_learning_rate = 0.001


class TransEModel(object):
    def __init__(self, config):
        batch_size = config.batch_size
        test_flag = config.test_flag
        margin = config.margin
        entity_embedding_size = config.embedding_size
        relation_embedding_size = config.relation_embedding_size
        entity_num = config.entity_num
        relation_num = config.relation_num
        base_learning_rate = config.base_learning_rate

        self.pos_h = tf.placeholder(tf.int32, [batch_size])
        self.pos_r = tf.placeholder(tf.int32, [batch_size])
        self.pos_e = tf.placeholder(tf.int32, [batch_size])

        self.neg_h = tf.placeholder(tf.int32, [batch_size])
        self.neg_r = tf.placeholder(tf.int32, [batch_size])
        self.neg_e = tf.placeholder(tf.int32, [batch_size])

        with tf.device("/cpu:0"):
            self.entity_embeddings = tf.get_variable("entity_embeddings", [entity_num, entity_embedding_size], dtype=tf.float32)
            self.relation_embeddings = tf.get_variable("relation_embeddings", [relation_num, relation_embedding_size], dtype=tf.float32)

            embedded_pos_h = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_h)
            embedded_pos_r = tf.nn.embedding_lookup(self.relation_embeddings, self.pos_r)
            embedded_pos_e = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_e)

            embedded_neg_h = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_h)
            embedded_neg_r = tf.nn.embedding_lookup(self.relation_embeddings, self.neg_r)
            embedded_neg_e = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_e)

        pos = tf.reduce_sum((embedded_pos_h + embedded_pos_r - embedded_pos_e) ** 2, 1, True)
        neg = tf.reduce_sum((embedded_neg_h + embedded_neg_r - embedded_neg_e) ** 2, 1, True)

        self.loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

        if not test_flag:
            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                base_learning_rate,
                global_step,
                300,
                0.98
            )

            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


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





if __name__ == '__main__':
    main()





