import torch.optim as optim
import random
import logging
import datetime
import os
from utility.parser import parse_args
from utility.batch_test import *
from utility.load_data import *
from tqdm import tqdm
from time import time
from copy import deepcopy
from model import DR
from utility.NumberKit import NumberKit

args = parse_args()
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list


if __name__ == '__main__':

    data_generator = Data(args)
    n_users = data_generator.n_users
    n_items = data_generator.n_items

    train_mat = data_generator.train_mat
    rows = train_mat.row
    cols = train_mat.col

    save_model_path = args.data_path + args.dataset + "/model_" + str(args.n_intents) + ".pth"
    _model = torch.load(save_model_path)
    _model.eval()
    _model.inference()

    usets = {}
    isets = {}

    for index in range(len(rows)):

        u1 = rows[index]
        i1 = cols[index]

        NumberKit.KeySetAddData(usets, u1, i1)
        NumberKit.KeySetAddData(isets, i1, u1)
    
    down_flag = [0, 10, 20, 30, 40]
    up_flag = [10, 20, 30, 40, 50]
    origin_test_set = data_generator.test_set

    user_test_sets = []
    for i in range(len(down_flag)):
        
        tus = {}
        df = down_flag[i]
        uf = up_flag[i]

        for j in list(origin_test_set.keys()):
            dj = len(usets[j]) 
            if dj >= df and dj < uf:
                tus[j] = origin_test_set[j]

        user_test_sets.append(tus)


    for index in range(len(user_test_sets)):

        print('user index:'+str(index))
        ts = user_test_sets[index]
        test_ret = eval_specified_test_set(_model, data_generator, eval(args.Ks), ts)
        perf_str = 'test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % \
                    (test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
        print(perf_str)
        print('********************')

    print('--------------------------------------------------------------------------------')


    item_test_sets = []
    for i in range(len(down_flag)):

        tus = {}
        df = down_flag[i]
        uf = up_flag[i]

        for j in list(origin_test_set.keys()):
            for k in origin_test_set[j]:

                dj = len(isets[k]) 
                if dj >= df and dj < uf:

                    if j not in tus.keys():
                        tus[j] = [k]
                    else:
                        tus[j].append(k)

        item_test_sets.append(tus)


    for index in range(len(item_test_sets)):

        print('item index:'+str(index))
        tu = item_test_sets[index]
        test_ret = eval_specified_test_set(_model, data_generator, eval(args.Ks), tu)
        perf_str = 'test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % \
                    (test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
        print(perf_str)
        print('********************')

    print('--------------------------------------------------------------------------------')