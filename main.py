import torch.optim as optim
import random
import logging
import datetime
import os
import math
from utility.parser import parse_args
from utility.batch_test import *
from utility.load_data import *
from tqdm import tqdm
from time import time
from copy import deepcopy
from model import DR
from model import AutoEncoder
from torch.utils.data import DataLoader
from dhg import Graph, BiGraph, Hypergraph
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utility.utils import *


import torch
import torch.optim as optim

args = parse_args()
# seed = args.seed
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
def init_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

seed = 42  # 可以是任何数字，确保每次运行都使用同一个种子
init_seed(seed)

# Function to check NaN and Inf
def check_nan(tensor, name="Tensor"):
    if isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            if torch.isnan(t).any():
                print(f"{name} {i} contains NaNs")
            if torch.isinf(t).any():
                print(f"{name} {i} contains Infs")
    else:
        if torch.isnan(tensor).any():
            print(f"{name} contains NaNs")
        if torch.isinf(tensor).any():
            print(f"{name} contains Infs")


def load_adjacency_list_data(adj_mat):
   
    coo_mat = adj_mat.tocoo()  
    all_h_list = list(coo_mat.row)  
    all_t_list = list(coo_mat.col)  
    all_v_list = list(coo_mat.data)  
    return all_h_list, all_t_list, all_v_list

# def init_seed(seed, reproducibility):
#     """Init random seed for random functions in numpy, torch, cuda and cudnn."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     if reproducibility:
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#     else:
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False

if __name__ == '__main__':

    """
    *********************************************************
    Prepare the log file
    """
    # init_seed(42, True)
    curr_time = datetime.datetime.now()
    if not os.path.exists('log'):
        os.mkdir('log')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler(
        'log/{}.log'.format(args.dataset), 'a', encoding='utf-8')
    logfile.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """
    *********************************************************
    Prepare the dataset
    """
    data_generator = Data(args)
    logger.info(data_generator.get_statistics())

    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    logger.info(args)
    print("************************************************************************************")

    config = dict()
    config['kHop'] = args.kHop
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['train_mat'] = data_generator.train_mat
    config['uu_mat'] = data_generator.uu_mat
    config['ii_mat'] = data_generator.ii_mat

    """
    *********************************************************
    Generate the adj matrix
    """
    plain_adj = data_generator.get_adj_mat()
    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)
    config['plain_adj'] = plain_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list

    config['uu_h_list'] = data_generator.uu_mat['row']
    config['uu_t_list'] = data_generator.uu_mat['col']
    config['uu_data'] = data_generator.uu_mat['data']

    config['ii_h_list'] = data_generator.ii_mat['row']
    config['ii_t_list'] = data_generator.ii_mat['col']
    config['ii_data'] = data_generator.ii_mat['data']

    config['u_v_info'] = data_generator.u_v_info
    config['v_u_info'] = data_generator.v_u_info

    config['intent_normalize'] = args.intent_normalize

    config['ui_bigraph'] = BiGraph.from_adj_list(data_generator.n_users, data_generator.n_items,
                                                 data_generator.adj_list, device=device)

    """
    *********************************************************
    Prepare the model and start training
    """
    social_edge_list = generate_edge_list(data_generator.social_adj)
    social_g = Graph(config['n_users'], social_edge_list)
    social_hg = Hypergraph.from_graph_kHop(social_g, k=config['kHop'], device=device)
    user_hg = Hypergraph.from_graph(social_g, device=device)

    h_edge = social_hg.e[0]
    config['edge_h_list'] = torch.concat((social_hg.v2e_src, social_hg.e2v_src + config['n_users']))
    config['edge_t_list'] = torch.concat((social_hg.e2v_src + config['n_users'], social_hg.v2e_src))

    config['H_shape'] = (config['n_users'] + len(h_edge), config['n_users'] + len(h_edge))
    config['h_edge'] = h_edge

    config['deg_v'] = social_hg.deg_v
    config['deg_e'] = social_hg.deg_e

    config['deg_v'].extend(social_hg.deg_e)
    config['deg_e'].extend(social_hg.deg_v)
    _model = DR(config, args).cuda()
    optimizer = optim.Adam(_model.parameters(), lr=args.lr, weight_decay=0.2)
    #optimizer = optim.RMSprop(_model.parameters(), lr=0.001, weight_decay=1e-4, momentum=0.9)

    print("Start Training")
    save_model_path = args.data_path + args.dataset + "/model_" + str(args.n_intents) + ".pth"

    best_score = 0
    best_epoch = 0
    best_result = None
    early_num = 0

    ui_data = [data_generator.n_users, data_generator.n_items, data_generator.adj_list]

    collator = Collator(data_generator.train_u_mat, ui_data)
    n_batch = math.ceil(len(data_generator.train_u_mat) / args.batch_size)

    train_loader = DataLoader(data_generator.train_u_mat, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(data_generator.test_u_mat, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.num_workers)

    for epoch in range(args.epoch):
        t1 = time()
        _model.train()
        loss, mf_loss, cen_loss, dis_loss, cl_loss, node_loss = 0., 0., 0., 0., 0., 0.
        for batch_idx, batch in enumerate(train_loader):
            trustor, pos_trustee, neg_trustee, pos_items, neg_items= batch
            batch_outputs = _model(trustor, pos_trustee, neg_trustee, social_hg, pos_items, neg_items)
            check_nan(batch_outputs, "Batch Outputs")
            # labels = labels.to(self.config.device)

            batch_mf_loss, batch_cen_loss, batch_dis_loss, batch_cl_loss, batch_node_loss = batch_outputs
            batch_loss = batch_mf_loss + batch_cen_loss + batch_dis_loss + batch_cl_loss + batch_node_loss
            check_nan(batch_loss, "Batch Loss")

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), max_norm=1.0)
            check_nan(batch_loss, "Batch Loss after backward")

            for name, param in _model.named_parameters():
                if param.grad is not None:
                    check_nan(param.grad, f"Gradient of {name}")

            optimizer.step()

            loss += batch_loss.item()
            mf_loss += batch_mf_loss.item()
            cen_loss += batch_cen_loss.item()
            dis_loss += batch_dis_loss.item()
            cl_loss += batch_cl_loss.item()
            node_loss += batch_node_loss



        perf_str = 'Epoch %2d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f]' % (
            epoch, time() - t1, loss, mf_loss, cen_loss, dis_loss, cl_loss, node_loss)
        print(perf_str)
        logger.info(perf_str)

        if epoch % args.show_step == 0 or epoch == args.epoch - 1:
            t2 = time()
            _model.eval()
            res_out = []
            labels_list = []
            for batch_idx, batch in enumerate(test_loader):
                trustor, trustee, labels = batch
                trustor = trustor.to(device)
                trustee = trustee.to(device)
                labels = labels.cpu().numpy()
                output = _model.predict(trustor, trustee)
                predicts = (output >= 0.5).cpu().numpy()
                res_out.extend(predicts)
                labels_list.extend(labels)
            acc = accuracy_score(labels_list, res_out)
            pre = precision_score(labels_list, res_out, zero_division=0)
            recall = recall_score(labels_list, res_out, zero_division=0)
            f1 = f1_score(labels_list, res_out, zero_division=0)
            auc = roc_auc_score(labels_list, res_out)

            perf_str = ('Epoch %2d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f + %.5f],'
                        ' recall=[%.4f], acc=[%.4f], pre=[%.4f], f1=[%.4f], auc=[%.4f]') % \
                       (epoch, t2 - t1, time() - t2, loss, mf_loss, cen_loss, dis_loss, cl_loss, node_loss,
                        recall, acc, pre, f1, auc)
            print(perf_str)
            logger.info(perf_str)

            score = acc + f1
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_result = loss
                early_num = 0
                best_acc = acc
                best_f1 = f1

                # if args.save_model:
                #     torch.save(_model, save_model_path)
            else:
                early_num = early_num + 1

            if early_num >= args.early_stop:
                break

        # best_result
        pref_str = 'Best Result at Epoch %2d: acc=[%.4f], f1=[%.4f]' % (best_epoch, best_acc, best_f1)
        print(pref_str)
        logger.info(pref_str)
