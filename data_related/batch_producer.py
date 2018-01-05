import tensorflow as tf
import numpy as np
from data_related.data_reader import *


def tfrecord_write(data, record_file):
    writer = tf.python_io.TFRecordWriter(record_file)
    for i in range(0, len(data)):
        example = tf.train.Example(features=tf.train.Features(feature={
            "head": tf.train.Feature(int64_list=tf.train.Int64List(value=[data[i][0]])),
            "relation": tf.train.Feature(int64_list=tf.train.Int64List(value=[data[i][2]])),
            "tail": tf.train.Feature(int64_list=tf.train.Int64List(value=[data[i][1]])),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def tfrecord_read(record_file):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([record_file])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "head": tf.FixedLenFeature([], tf.int64),
            "relation": tf.FixedLenFeature([], tf.int64),
            "tail": tf.FixedLenFeature([], tf.int64),
        }
    )

    head = tf.cast(features["head"], tf.int32)
    relation = tf.cast(features["relation"], tf.int32)
    tail = tf.cast(features["tail"], tf.int32)

    return head, relation, tail


def get_batch():

    return


