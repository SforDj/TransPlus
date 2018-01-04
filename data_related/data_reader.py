import numpy as np


def read_entity2id(file):
    entity2id_dic = dict()
    with open(file, 'r', encoding='UTF-8') as in_stream:
        size = int(in_stream.readline().strip())
        while True:
            line = in_stream.readline().strip()
            if line == "":
                break
            line_arr = line.split("\t")
            entity2id_dic[line_arr[0]] = int(line_arr[1])

    return entity2id_dic, size


def read_relation2id(file):
    relation2id_dic = dict()
    with open(file, 'r', encoding='UTF-8') as in_stream:
        size = int(in_stream.readline().strip())
        while True:
            line = in_stream.readline().strip()
            if line == "":
                break
            line_arr = line.split("\t")
            relation2id_dic[line_arr[0]] = int(line_arr[1])

    return relation2id_dic, size


def read_triple(file):
    triple = []
    with open(file, 'r', encoding='UTF-8') as in_stream:
        size = int(in_stream.readline().strip())
        while True:
            line = in_stream.readline().strip()
            if line == "":
                break
            line_arr = line.split(" ")
            triple.append(int(line_arr[0]))
            triple.append(int(line_arr[1]))
            triple.append(int(line_arr[2]))

    triple = np.reshape(triple, [size, 3])
    return triple

