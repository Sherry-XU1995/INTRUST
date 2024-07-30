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

    ua_embedding = _model.ua_embedding
    ia_embedding = _model.ia_embedding
    ua_embedding = torch.nn.functional.normalize(ua_embedding)
    ia_embedding = torch.nn.functional.normalize(ia_embedding)

    um_embedding = torch.mean(ua_embedding, dim=0)
    madu = torch.mean(torch.abs(torch.matmul(ua_embedding, um_embedding.T)), dim=0)
    print('u mad:' + str(madu))

    im_embedding = torch.mean(ia_embedding, dim=0)
    madi = torch.mean(torch.abs(torch.matmul(ia_embedding, im_embedding.T)), dim=0)
    print('i mad:' + str(madi))