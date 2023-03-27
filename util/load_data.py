from re import split
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def load_data_set(file):
    data = []
    with open(file) as f:
        for line in f:
            items = split(' ', line.strip())
            user_id = int(items[0])
            item_id = int(items[1])
            weight = int(items[2])
            data.append([user_id, item_id, weight])
    return data


class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training
        self.test_data = test  # can also be validation set if the input is for validation
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.train_item_id = []
        self.train_item_idx = []
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.test_users_idx = []
        self.test_items_id = []
        self.test_items_idx = []
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.__normalize_graph_mat(self.ui_adj)

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry   #id
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
                # userList.append
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        self.train_item_id = [list(self.training_set_u[u]) for u in self.training_set_u.keys()]
        self.train_item_idx = [list(map(self.item.get, itemlist)) for itemlist in self.train_item_id]

        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user or item not in self.item:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)
        self.test_users_idx = [self.user[u] for u in self.test_set.keys()]
        self.test_items_id = [list(self.test_set[i]) for i in self.test_set.keys()]
        self.test_items_idx = [list(map(self.item.get, itemlist)) for itemlist in self.test_items_id]

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        '''
        return a sparse adjacency matrix with the shape (user number + item number, user number + item number)
        '''
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes),dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat


    def __normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]