import numpy as np
import os
from os import listdir
from os.path import isfile, join
import collections
import re
from tqdm import trange
from tqdm import *
import random
# import pickle
import pyarrow


def load_graphs_from_file(file_name):
    data_list = []
    edge_list = []
    target_list = []

    with open(file_name, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                data_list.append([edge_list, target_list])
                edge_list = []
                target_list = []
            else:
                digits = []
                line_tokens = line.split(" ")

                if line_tokens[0] == "?":
                    for i in range(1, len(line_tokens)):

                        digits.append(int(line_tokens[i]))
                    target_list.append(digits)
                else:
                    for i in range(len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    edge_list.append(digits)
    # print(data_list)
    return data_list


def lookup_vector(node_type, embeddings):
    # look_up_vector = embed_loopkup[node_type]

    nodes.append(embeddings[int(n)])


def load_program_graphs_from_directory(directory, is_train=True, n_classes=3, data_percentage=1.0):

    node_id_data_list = []
    node_type_data_list = []
    if is_train == True:
        dir_path = os.path.join(directory, "train")
    else:
        dir_path = os.path.join(directory, "test")
    filenames = []
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            filenames.append(f)
    int_filenames = [int(re.search('_(.*).txt', x).group(1))
                     for x in filenames]
    ordered_filenames = sorted(int_filenames)
    lookup = {}
    for i in range(1, 1+len(ordered_filenames)):
        if is_train == True:
            lookup[i] = join(dir_path, "train_%s.txt" %
                             str(ordered_filenames[i-1]))
        else:
            lookup[i] = join(dir_path, "test_%s.txt" %
                             str(ordered_filenames[i-1]))
    for i in trange(1, 1+n_classes):
        path = lookup[i]
        print(path)
        label = i
        node_id_data_list_class_i = []
        node_type_data_list_class_i = []
        node_id_edge_list_class_i = []
        node_type_edge_list_class_i = []
        target_list_class_i = []

        print("--------------------------")
        with open(path, 'r') as f:
            for line in f:
                if len(line.strip()) == 0:
                    # print(edge_list_class_i)
                    node_id_data_list_class_i.append(
                        [node_id_edge_list_class_i, target_list_class_i])
                    node_type_data_list_class_i.append(
                        [node_type_edge_list_class_i, target_list_class_i])
                    node_id_edge_list_class_i = []
                    node_type_edge_list_class_i = []
                    target_list_class_i = []
                else:
                    node_id_digits = []
                    node_type_digits = []
                    line_tokens = line.split(" ")

                    if line_tokens[0] == "?":

                        target_list_class_i.append([label])
                    else:

                        for j in range(len(line_tokens)):
                            if "," in line_tokens[j]:
                                splits = line_tokens[j].split(",")
                                node_id = splits[0]
                                node_type = splits[1]
                                node_id_digits.append(int(node_id))
                                node_type_digits.append(int(node_type))
                            else:
                                node_id_digits.append(int(line_tokens[j]))
                                node_type_digits.append(int(line_tokens[j]))

                        node_id_edge_list_class_i.append(node_id_digits)
                        node_type_edge_list_class_i.append(node_type_digits)

        # if data_percentage < 1.0:
        #     print("Cutting down " + str(data_percentage) + " of all data......")
        #     slicing = int(len(node_id_data_list_class_i)*data_percentage)
        #     print("Remaining data : " + str(slicing) + "......")
        #     node_id_data_list_class_i = node_id_data_list_class_i[:slicing]

        node_id_data_list.extend(node_id_data_list_class_i)
        node_type_data_list.extend(node_type_data_list_class_i)

    # print(node_embeddings)
    return node_id_data_list, node_type_data_list


def find_max_edge_id(data_list):
    max_edge_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[1] > max_edge_id:
                max_edge_id = item[1]
    return max_edge_id


def find_max_node_id(data_list):
    max_node_id = 0
    for data in data_list:
        edges = data[0]
        for item in edges:
            if item[0] > max_node_id:
                max_node_id = item[0]
            if item[2] > max_node_id:
                max_node_id = item[2]
    return max_node_id
    # return 48


def find_max_task_id(data_list):
    max_node_id = 0
    for data in data_list:
        targe = data[1]
        for item in targe:
            if item[0] > max_node_id:
                max_node_id = item[0]
    return max_node_id


def split_set(data_list, num):
    n_examples = len(data_list)
    idx = range(n_examples)
    train = idx[:num]
    val = idx[-num:]
    return np.array(data_list)[train], np.array(data_list)[val]


def split_set_by_percentage(data_list, percentage):
    n_examples = len(data_list)
    train_num = int(n_examples * percentage)

    idx = range(n_examples)
    train = idx[:train_num]
    val = idx[train_num:n_examples]
    return np.array(data_list)[train], np.array(data_list)[val]


def convert_program_data(data_list, n_annotation_dim, n_nodes):
    # n_nodes = find_max_node_id(data_list)
    class_data_list = []

    for item in data_list:
        edge_list = item[0]
        target_list = item[1]
        for target in target_list:
            task_type = target[0]
            task_output = target[-1]
            annotation = np.zeros([n_nodes, n_annotation_dim])
            for edge in edge_list:
                src_idx = edge[0]

                if src_idx < len(annotation):
                    annotation[src_idx-1][0] = 1

            class_data_list.append([edge_list, annotation, task_output])
    return class_data_list


def convert_program_data_into_group(data_list, n_annotation_dim, n_nodes, n_classes):
    class_data_list = []

    for i in range(n_classes):
        class_data_list.append([])

    for item in data_list:
        edge_list = item[0]
        target_list = item[1]
        for target in target_list:
            class_output = target[-1]
            annotation = np.zeros([n_nodes, n_annotation_dim])
            for edge in edge_list:
                src_idx = edge[0]
                if src_idx < len(annotation):
                    annotation[src_idx-1][0] = 1
            # print(class_output)
            class_data_list[class_output -
                            1].append([edge_list, annotation, class_output])
    return class_data_list


def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])

    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]

        if tgt_idx < len(a):
            a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        if src_idx < len(a):
            a[src_idx-1][(e_type - 1 + n_edge_types)
                         * n_nodes + tgt_idx - 1] = 1
    return a


def create_embedding_matrix(node_id_edges, node_type_edges, n_nodes, pretrained_embeddings):
    a = np.zeros([n_nodes, 30])
    # print(a.shape)
    for i in range(len(node_id_edges)):
        node_type = node_type_edges[i][0]
        # print(node_type)
        src_idx = node_id_edges[i][0]
        a[src_idx-1] = pretrained_embeddings[int(node_type)]
    return a


class FFMQProgramData():

    def __init__(self, size_vocabulary, embeddings, path, is_train, n_classes=3, data_percentage=1.0):
        base_name = os.path.basename(path)
        if is_train:
            saved_input_filename = "%s/%s-%d-train.pkl" % (
                path, base_name, n_classes)
        else:
            saved_input_filename = "%s/%s-%d-test.pkl" % (
                path, base_name, n_classes)
        if os.path.exists(saved_input_filename):
            input_file = open(saved_input_filename, 'rb')
            buf = input_file.read()
            all_data_node_id, all_data_node_type = pyarrow.deserialize(buf)
            input_file.close()
        else:
            all_data_node_id, all_data_node_type = load_program_graphs_from_directory(
                path, is_train, n_classes, data_percentage)
            all_data_node_id = np.array(all_data_node_id)[
                0:len(all_data_node_id)]
            all_data_node_type = np.array(all_data_node_type)[
                0:len(all_data_node_type)]
            buf = pyarrow.serialize(
                (all_data_node_id, all_data_node_type)).to_buffer()
            out = pyarrow.OSFile(saved_input_filename, 'wb')
            out.write(buf)
            out.close()

        self.pretrained_embeddings = embeddings
        # print(all_data_node_id)
        if is_train == True:
            print("Number of all training data : " + str(len(all_data_node_id)))
        else:
            print("Number of all testing data : " + str(len(all_data_node_id)))
        self.n_edge_types = find_max_edge_id(all_data_node_id)
        # print("Edge types : " + str(self.n_edge_types))
        max_node_id = find_max_node_id(all_data_node_id)
        max_node_type = find_max_node_id(all_data_node_type)
        print("Max node id : " + str(max_node_id))
        print("Max node type : " + str(max_node_type))
        # self.n_node = size_vocabulary
        self.n_node_by_id = max_node_id
        self.n_node = max_node_id  # set n_node = n_node_by_id
        self.n_node_by_type = max_node_type

        all_data_node_id = convert_program_data(
            all_data_node_id, 1, self.n_node_by_id)
        all_data_node_type = convert_program_data(
            all_data_node_type, 1, self.n_node_by_type)

        self.all_data_node_id = all_data_node_id
        self.all_data_node_type = all_data_node_type

        self.data = all_data_node_id
        # self.embedding_matrix = create_embedding_matrix(self.data, self.n_node,  self.pretrained_embeddings)

    def __getitem__(self, index):

        am = create_adjacency_matrix(
            self.data[index][0], self.n_node, self.n_edge_types)
        embedding_matrix = create_embedding_matrix(
            self.all_data_node_id[index][0], self.all_data_node_type[index][0], self.n_node,  self.pretrained_embeddings)
        # embedding = embeddings[int(node_type)]
        # annotation = self.data[index][1]
        target = self.data[index][2] - 1
        return am, embedding_matrix, target

    def __len__(self):
        return len(self.data)


class VerumProgramData():

    def __init__(self, size_vocabulary, left_path, right_path, is_train, loss, n_classes=3, data_percentage=1):

        self.loss = loss
        left_all_data = load_program_graphs_from_directory(
            left_path, is_train, n_classes, data_percentage)
        right_all_data = load_program_graphs_from_directory(
            right_path, is_train, n_classes, data_percentage)

        left_all_data = np.array(left_all_data)[0:len(left_all_data)]
        right_all_data = np.array(right_all_data)[0:len(right_all_data)]

        if is_train == True:
            print("Number of all left training data : " + str(len(left_all_data)))
            print("Number of all right training data : " +
                  str(len(right_all_data)))
        else:
            print("Number of all left testing data : " + str(len(left_all_data)))
            print("Number of all right testing data : " +
                  str(len(right_all_data)))

        self.n_edge_types = find_max_edge_id(left_all_data)
        self.n_node = size_vocabulary
        max_left_node = find_max_node_id(left_all_data)
        max_right_node = find_max_node_id(right_all_data)

        left_all_data_by_classes = convert_program_data_into_group(
            left_all_data, 1, self.n_node, n_classes)

        right_all_data_by_classes = convert_program_data_into_group(
            right_all_data, 1, self.n_node, n_classes)

        pairs_1 = []
        pairs_0 = []

        for i, left_class in tqdm(enumerate(left_all_data_by_classes)):
            right_class = right_all_data_by_classes[i]

            remaining_right_class = []

            for j, other_right_class in enumerate(right_all_data_by_classes):
                if j != i:
                    remaining_right_class.extend(other_right_class)

            if len(left_class) > len(right_class):
                left_class = left_class[:len(right_class)]

            for k, left_data_point in enumerate(left_class):
                righ_data_point = right_class[k]
                pairs_1.append((left_data_point, righ_data_point))
                pairs_0.append(
                    (left_data_point, random.choice(remaining_right_class)))

        data = []
        data.extend(pairs_1)
        data.extend(pairs_0)
        random.shuffle(data)
        self.data = data

    def __getitem__(self, index):

        left_data_point = self.data[index][0]
        right_data_point = self.data[index][1]

        left_am = create_adjacency_matrix(
            left_data_point[0], self.n_node, self.n_edge_types)
        right_am = create_adjacency_matrix(
            right_data_point[0], self.n_node, self.n_edge_types)

        left_annotation = left_data_point[1]
        right_annotation = right_data_point[1]

        if left_data_point[2] == right_data_point[2]:
            target = 1.0
        else:
            target = 0.0

        if self.loss == 0:
            target = int(target)

        return (left_am, right_am), target

    def __len__(self):
        return len(self.data)
