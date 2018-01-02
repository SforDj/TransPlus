import tensorflow as tf


class Config(object):
    def __init__(self):
        self.test_flag = False
        self.embedding_size = 128
        self.margin = 1.0
        self.train_times = 500
        self.relation_num = 0
        self.entity_num = 0
        self.batch_size = 32


class TransEModel(object):
