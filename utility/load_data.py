import torch
import random
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp
from pathlib import Path
from dhg.datapipe import load_from_txt, load_from_pickle


class Data(object):
    def __init__(self, args):
        # cur_path = os.getcwd()
        pro_path = Path(__file__).parent.parent
        data_path = Path(pro_path, 'data')
        self.path = Path(data_path, args.dataset)
        self.n_batch = args.n_batch
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.sample_num = args.sample_num

        try:
            # train_file = self.path + '/rating.pkl'
            # train_u_files = self.path + '/train_mask.pkl'
            # test_u_file = self.path + '/test_mask.pkl'
            # social_adj_file = self.path + '/social_adj.pkl'
            # with open(train_file, 'rb') as f:
            #     train_mat = pickle.load(f)
            # with open(train_u_files, 'rb') as f:
            #     train_u_mat = pickle.load(f)
            # with open(test_u_file, 'rb') as f:
            #     test_u_mat = pickle.load(f)
            # with open(social_adj_file, 'rb') as f:
            #     social_adj = pickle.load(f)
            # train_mat = load_from_pickle(Path(self.path, 'rating.pkl'))
            train_u_mat = load_from_pickle(Path(self.path, 'train_mask.pkl'))
            test_u_mat = load_from_pickle(Path(self.path, 'test_mask.pkl'))
            self.social_adj = load_from_pickle(Path(self.path, 'social_adj.pkl'))
            bi_adj_list = load_from_txt(Path(self.path, 'adj_list.txt'), dtype="int", sep=" ")
            u_v_info = load_from_pickle(Path(self.path, 'u_v_info.pkl'))
            v_u_info = load_from_pickle(Path(self.path, 'v_u_info.pkl'))
            self.u_v_info = u_v_info
            self.v_u_info = v_u_info
        except Exception as e:
            print(e)
            print("The dataset not exist")
            # print("Try an alternative way of reading the data.")
            # train_file = self.path + '/train_index.pkl'
            # test_file = self.path + '/test_index.pkl'
            # with open(train_file, 'rb') as f:
            #     train_index = pickle.load(f)
            # with open(test_file, 'rb') as f:
            #     test_index = pickle.load(f)
            # train_row, train_col = train_index[0], train_index[1]
            # n_user = max(train_row) + 1
            # n_item = max(train_col) + 1
            # train_mat = sp.coo_matrix((np.ones(len(train_row)), (train_row, train_col)), shape=[n_user, n_item])
            # test_row, test_col = test_index[0], test_index[1]
            # test_mat = sp.coo_matrix((np.ones(len(test_row)), (test_row, test_col)), shape=[n_user, n_item])

        try:
            # kitlee
            # uu_file = self.path + '/uu' + str(args.ui_k) + '.pkl'
            # ii_file = self.path + '/ii' + str(args.ui_k) + '.pkl'
            # with open(uu_file, 'rb') as f:
            #     uu_mat = pickle.load(f)
            # with open(ii_file, 'rb') as f:
            #     ii_mat = pickle.load(f)

            uu_mat = load_from_pickle(Path(self.path, 'uu5.pkl'))
            ii_mat = load_from_pickle(Path(self.path, 'ii5.pkl'))
            self.uu_mat = uu_mat
            self.ii_mat = ii_mat
        except Exception as e:
            pass

        # kitlee
        # train_mat = np.sort(train_mat, axis=0)
        # train_mat = train_mat[train_mat[:,2] > 0]
        # train_mat = np.unique(train_mat, axis=0)
        # train_row, train_col, train_data = train_mat[:, 0]-1, train_mat[:, 1], train_mat[:, 2]
        # # train_data = train_mat[train_mat.nonzero()]
        # sp_train_mat = sp.coo_matrix((train_data, (train_row, train_col)), shape=train_mat.shape)
        # get number of users and items
        # build the
        self.n_users, self.n_items = len(u_v_info), len(v_u_info)
        train_mat = generate_train_mat(self.n_users, self.n_items, bi_adj_list)
        self.train_mat = train_mat

        self.n_train, self.n_test = len(train_u_mat), len(test_u_mat)

        self.print_statistics()

        self.R = train_mat.todok()
        self.train_items, self.test_set = {}, {}
        train_uid, train_iid = train_mat.row, train_mat.col

        self.train_u_mat = train_u_mat
        self.test_u_mat = test_u_mat
        self.adj_list = bi_adj_list
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid)
        # test_uid, test_iid = test_mat.row, test_mat.col
        # for i in range(len(test_uid)):
        #     uid = test_uid[i]
        #     iid = test_iid[i]
        #     if uid not in self.test_set:
        #         self.test_set[uid] = [iid]
        #     else:
        #         self.test_set[uid].append(iid)

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def get_uu_adj_mat(self):
        rows = self.uu_mat['row']
        cols = self.uu_mat['col']
        data = self.uu_mat['data']
        adj_mat = sp.coo_matrix((data, (rows, cols)), shape=[self.n_users, self.n_users]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        return adj_mat.tocsr()

    def get_ii_adj_mat(self):
        rows = self.ii_mat['row']
        cols = self.ii_mat['col']
        data = self.ii_mat['data']
        adj_mat = sp.coo_matrix((data, (rows, cols)), shape=[self.n_items, self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        return adj_mat.tocsr()

    def uniform_sample(self):
        users = np.random.randint(0, self.n_users, int(self.n_batch * self.batch_size))
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            pos_for_user = self.train_items[user]
            pos_index = np.random.randint(0, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if self.R[user, neg_item] == 1:
                    continue
                else:
                    break
            train_data.append([user, pos_item, neg_item])
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta


def generate_train_mat(row, col, adj_list):
    e_list = []
    for line in adj_list:
        if len(line) <= 1:
            continue
        u_idx = line[0]
        e_list.extend([[u_idx, v_idx] for v_idx in line[1:]])
    adj_shape = (row, col)
    # TODO: can extract the user and item weights distribution
    e_arr = np.unique(np.sort(np.array(e_list), axis=0), axis=0)

    adj_data = np.ones(e_arr.shape[0], dtype=np.int32)
    adj_row, adj_col = e_arr[:, 0], e_arr[:, 1]
    adj_matrix = sp.coo_matrix((adj_data, (adj_row, adj_col)), shape=adj_shape)

    return adj_matrix


class Collator:
    def __init__(self,  data, ui_data):
        self.n_users = ui_data[0]
        self.n_items = ui_data[1]
        self.pos_pairs = [_ for _ in data if _[2] == 1]
        self.neg_pairs = [_ for _ in data if _[2] == 0]
        self.trustor_neg_map = {}
        for pair in self.neg_pairs:
            self.trustor_neg_map.setdefault(pair[0], []).append(pair[1])

        adj_list = ui_data[2]
        self.train_items = dict()
        for index in range(len(adj_list)):
            self.train_items[adj_list[index][0]] = adj_list[index][1:]

    def __call__(self, data):
        trustor = []
        pos_trustee = []
        neg_trustee = []

        for pair in data:
            trustor.append(pair[0])
            pos_trustee.append(pair[1])
            neg_trustee.append(random.choice(self.trustor_neg_map[pair[0]]))

        pos_items, neg_items = self.uniform_sample(trustor)
        trustor = torch.IntTensor(trustor)
        pos_trustee = torch.IntTensor(pos_trustee)
        neg_trustee = torch.IntTensor(neg_trustee)
        pos_items_data = torch.IntTensor(pos_items)
        neg_items_data = torch.IntTensor(neg_items)
        return trustor, pos_trustee, neg_trustee, pos_items_data, neg_items_data

    def uniform_sample(self, users):
        train_pos_item = []
        train_neg_item = []
        for i, user in enumerate(users):
            pos_for_user = self.train_items[user]
            if len(pos_for_user) > 0:
                pos_index = np.random.randint(0, len(pos_for_user))
                pos_item = pos_for_user[pos_index]
                while True:
                    neg_item = np.random.randint(0, self.n_items)
                    if neg_item in pos_for_user:
                        continue
                    else:
                        break
            else:

                pos_item = user
                neg_item = 0

            train_pos_item.append(pos_item)
            train_neg_item.append(neg_item)
        return train_pos_item, train_neg_item
