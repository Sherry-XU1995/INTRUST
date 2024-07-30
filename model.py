import torch
from torch.nn import LayerNorm
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import dhg
from dhg import BiGraph
from dhg.nn import UniGINConv, HGNNPConv, UniGCNConv, UniGATConv, UniSAGEConv, HyperGCNConv, GCNConv, GATConv
from typing import Optional
from torch import Tensor


def check_nan(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"{name} contains NaNs")
    if torch.isinf(tensor).any():
        print(f"{name} contains Infs")

class DR(nn.Module):
    def __init__(self, data_config, args):
        super(DR, self).__init__()

        self.hyper_x = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.intent_normalize = data_config['intent_normalize']

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).to(self.device)
        self.D_indices = torch.tensor(
            [list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).to(
            self.device)
        self.all_h_list = torch.LongTensor(self.all_h_list).to(self.device)
        self.all_t_list = torch.LongTensor(self.all_t_list).to(self.device)
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.n_intents = args.n_intents
        self.temp = args.temp
        self.hidden = args.hidden_dim
        self.conv_name = args.conv_name
        self.bias = args.bias
        self.out_dim = args.output_dim


        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.cen_reg = args.cen_reg
        self.cen_dis = args.cen_dis
        self.ssl_reg = args.ssl_reg
        self.tau = args.tau

        """
        *********************************************************
        user_item interaction
        """
        self.n_trains = len(data_config['train_mat'].row)
        user_ut = []
        train_ut = []
        for index, value in enumerate(data_config['train_mat'].row):
            user_ut.append(value)
            train_ut.append(index)

        item_it = []
        train_it = []
        for index, value in enumerate(data_config['train_mat'].col):
            item_it.append(value)
            train_it.append(index)

        self.UT_indices = torch.tensor([user_ut, train_ut], dtype=torch.long).to(self.device)
        self.TU_indices = torch.tensor([train_ut, user_ut], dtype=torch.long).to(self.device)
        self.UT_values = torch.tensor([1 for _ in range(self.n_trains)], dtype=torch.long).to(self.device)

        self.IT_indices = torch.tensor([item_it, train_it], dtype=torch.long).to(self.device)
        self.TI_indices = torch.tensor([train_it, item_it], dtype=torch.long).to(self.device)
        self.IT_values = torch.tensor([1 for _ in range(self.n_trains)], dtype=torch.long).to(self.device)

        self.h_list = torch.LongTensor(data_config['train_mat'].row).to(self.device)
        self.t_list = torch.LongTensor(data_config['train_mat'].col).to(self.device)

        """
        *********************************************************
        user_user interaction and item_item interaction
        """
        self.uu_h_list = torch.LongTensor(data_config['uu_h_list']).to(self.device)
        self.uu_t_list = torch.LongTensor(data_config['uu_t_list']).to(self.device)
        self.uu_data = torch.FloatTensor(data_config['uu_data']).to(self.device)
        self.uu_indices = torch.tensor([data_config['uu_h_list'], data_config['uu_t_list']], dtype=torch.long).to(self.device)
        self.uu_data = self.normalization_indices_values(self.uu_h_list, self.uu_t_list, self.uu_data, (self.n_users, self.n_users))

        self.ii_h_list = torch.LongTensor(data_config['ii_h_list']).to(self.device)
        self.ii_t_list = torch.LongTensor(data_config['ii_t_list']).to(self.device)
        self.ii_data = torch.FloatTensor(data_config['ii_data']).to(self.device)
        self.ii_indices = torch.tensor([data_config['ii_h_list'], data_config['ii_t_list']], dtype=torch.long).to(self.device)
        self.ii_data = self.normalization_indices_values(self.ii_h_list, self.ii_t_list, self.ii_data, (self.n_items, self.n_items))
        #


        """
        *********************************************************
        Create hypergraph Parameters 
        """

        self.h_edge = data_config['h_edge']
        self.n_h_edge = len(self.h_edge)
        self.H_shape = data_config['H_shape']
        self.hu_h_list = data_config['edge_h_list']
        self.hu_t_list = data_config['edge_t_list']

        self.h_deg_v_list = torch.tensor(data_config['deg_v'], dtype=torch.float32).to(self.device)
        self.h_deg_e_list = torch.tensor(data_config['deg_e'], dtype=torch.float32).to(self.device)
        self.deg_indices = torch.tensor(
            [list(range(self.n_users + self.n_h_edge)), list(range(self.n_users + self.n_h_edge))],
            dtype=torch.long).to(self.device)

        self.hu_h_indices = self.hu_h_list.clone().detach().to(self.device)
        self.hu_t_indices = self.hu_t_list.clone().detach().to(self.device)
        self.hu_indices = torch.stack((self.hu_h_indices, self.hu_t_indices))

        self.H_indices, self.H_values = self._cal_sparse_ind_matrix()

        """
        *********************************************************
        Create Model Parameters
        """
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        # This hyper_edge for HGNNP Conv
        self.hyper_edge_embedding = nn.Embedding(self.n_users, self.n_h_edge)
        # This hyperedge for intent
        self.hyperedge_embedding = nn.Embedding(self.n_h_edge, self.emb_dim)

        _intents = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_intents)
        self.intents = torch.nn.Parameter(_intents, requires_grad=True)

        _user_intents = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intents)
        self.user_intents = torch.nn.Parameter(_user_intents, requires_grad=True)

        _hg_intents = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_hg_intents)
        self.hg_intents = torch.nn.Parameter(_hg_intents, requires_grad=True)
        """
        *********************************************************
        Create light gcn layer
        """
        self.bi_graph = data_config['ui_bigraph']
        self.bi_graph = self.bi_graph.to(self.device)
        self.conv_list = {
            'hgnnp': HGNNPConv,
            'uingcn': UniGCNConv,
            'unigat': UniGATConv,
            'unisage': UniSAGEConv,
            'unigin': UniGINConv,
            'hypergcn': HyperGCNConv,
            'gcn': GCNConv,
            'gat': GATConv
        }

        self.light_layer = LightGCN(self.n_users, self.n_items, self.emb_dim)
        # self.Conv_Layer = self.conv_list[self.conv_name](self.emb_dim, self.hidden, bias=self.bias)
        self.Conv_Layer = self.conv_list[self.conv_name](self.emb_dim, self.emb_dim, bias=self.bias)
        self.Trustor_UniGAT_Layer = UniGATConv(self.emb_dim, self.hidden, self.bias)
        self.Trustee_UniGAT_Layer = UniGATConv(self.emb_dim, self.hidden, self.bias)
        self.act = nn.ReLU()
        self.theta = nn.Linear(self.hidden, self.out_dim, bias=self.bias)
        self.edge_theta = nn.Linear(self.n_h_edge, self.emb_dim, bias=self.bias)
        # self.linear_layer = nn.Linear(self.emb_dim * 2, self.emb_dim)

        self.mlp = MLP(input_dim=self.emb_dim*3, output_dim=self.emb_dim, hidden_size=(self.hidden, self.hidden))
        """
        *********************************************************
        Initialize Weights
        """
        self._init_weight()

        self.softplus = nn.Softplus(beta=0.5, threshold=20)

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.hyper_edge_embedding.weight)
        nn.init.xavier_normal_(self.hyperedge_embedding.weight)

    def _cal_sparse_adj(self):

        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).to(self.device)

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values,
                                             sparse_sizes=self.A_in_shape).to(self.device)
        D_values = (A_tensor.sum(dim=1) + 1e-12).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values,
                                                  self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0],
                                                  self.A_in_shape[1], self.A_in_shape[1])
        check_nan(G_values, "G_values after computation")

        return G_indices, G_values

    def _cal_sparse_ind_matrix(self):
        # H = D_v^{\frac{-1}{2}}HD_e^{-1}D_v^{\frac{-1}{2}}
        E_values = self.h_deg_e_list.pow(-1)
        D_values = self.h_deg_v_list.pow(-0.5)
        H_values = torch.ones(size=(self.hu_h_indices.shape[0], 1)).view(-1).to(self.device)

        G_indices, G_values = torch_sparse.spspmm(self.deg_indices, D_values, self.hu_indices, H_values,
                                                  self.H_shape[0], self.H_shape[1], self.H_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.deg_indices, E_values,
                                                  self.H_shape[0], self.H_shape[1], self.H_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.deg_indices, D_values,
                                                  self.H_shape[0], self.H_shape[1], self.H_shape[1])

        return G_indices, G_values

    def normalization_indices_values(self, h_list, t_list, value, shape):
        A_tensor = torch_sparse.SparseTensor(row=h_list, col=t_list, value=value, sparse_sizes=shape)
        D_scores_inv = (A_tensor.sum(dim=1) + 1e-12).pow(-1).view(-1)
        return D_scores_inv[h_list] * value

    def inference(self):

        # base_layer_embeddings = torch.cat([self.user_embedding.weight + self.hyper_x,
        #                                    self.item_embedding.weight + self.i_embs], dim=0)
        base_layer_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        check_nan(base_layer_embeddings, "Base Layer Embeddings")
        all_embeddings = [base_layer_embeddings]

        base_embeddings = []
        gnn_embeddings = []
        hio_embeddings = []
        fdr_embeddings = []  # 存储每一层的First Order Disentangled Representations

        for i in range(self.n_layers):
            # 图基础信息传递：使用图卷积更新嵌入
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0],
                                                     self.A_in_shape[1], all_embeddings[i])

            check_nan(gnn_layer_embeddings, f"GNN Layer Embeddings {i}")
            gnn_embeddings.append(gnn_layer_embeddings)
            # 高阶交互处理：用户-用户和物品-物品间的交互

            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)
            uu_embeddings = torch_sparse.spmm(self.uu_indices, self.uu_data, self.n_users, self.n_users, u_embeddings)
            ii_embeddings = torch_sparse.spmm(self.ii_indices, self.ii_data, self.n_items, self.n_items, i_embeddings)
            hio_layer_embeddings = torch.cat((uu_embeddings, ii_embeddings), dim=0)

            check_nan(hio_layer_embeddings, "HIO Layer Embeddings")
            hio_embeddings.append(hio_layer_embeddings)
            # 检查和处理非法值
            has_nan = torch.isnan(gnn_layer_embeddings).any()
            has_inf = torch.isinf(gnn_layer_embeddings).any()
            if has_nan or has_inf:
                gnn_layer_embeddings[torch.isnan(gnn_layer_embeddings)] = 0  # 将 nan 值替换为 0 或其他合适的值
                gnn_layer_embeddings[torch.isinf(gnn_layer_embeddings)] = 0  # 将 inf 值替换为 0 或其他合适的值

            #First Order Disentangled Representation
            head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.h_list)
            tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.t_list)
            edge_distributions = torch.softmax((head_embeddings * tail_embeddings) @ self.intents, dim=1)
            edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
            edge_alpha = edge_alpha.flatten()

            epsilon = 1e-12
            user_intents = torch_sparse.spmm(self.UT_indices, edge_alpha, self.n_users, self.n_trains,
                                             edge_distributions)
            user_intents = torch.div(torch.ones_like(user_intents), user_intents + epsilon)
            edge_user_intents = torch_sparse.spmm(self.TU_indices, edge_alpha, self.n_trains, self.n_users,
                                                  user_intents)
            edge_user_distribution = edge_distributions * edge_user_intents
            edge_user_distribution = torch.mean(edge_user_distribution, dim=1, keepdim=False)

            item_intents = torch_sparse.spmm(self.IT_indices, edge_alpha, self.n_items, self.n_trains,
                                             edge_distributions)
            item_intents = torch.div(torch.ones_like(item_intents), item_intents + epsilon)
            edge_item_intents = torch_sparse.spmm(self.TI_indices, edge_alpha, self.n_trains, self.n_items,
                                                  item_intents)
            edge_item_distribution = edge_distributions * edge_item_intents
            edge_item_distribution = torch.mean(edge_item_distribution, dim=1, keepdim=False)

            edge_distributions = torch.cat((edge_user_distribution, edge_item_distribution), dim=0)
            A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_distributions,
                                                 sparse_sizes=self.A_in_shape).cuda()
            D_scores_inv = A_tensor.sum(dim=1).nan_to_num(0, 0, 0).pow(-1).nan_to_num(0, 0, 0).view(-1)
            edge_distributions = D_scores_inv[self.all_h_list] * edge_distributions
            G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
            fdr_layer_embeddings = torch_sparse.spmm(G_indices, edge_distributions, self.A_in_shape[0],
                                                     self.A_in_shape[1], gnn_layer_embeddings)
            fdr_layer_embeddings = torch.nn.functional.normalize(fdr_layer_embeddings, dim=1, eps=1e-8)
            fdr_embeddings.append(fdr_layer_embeddings)

            # 更新当前层的嵌入，准备传递给下一层
            current_layer_embeddings = all_embeddings[
                                           i] + gnn_layer_embeddings + hio_layer_embeddings + fdr_layer_embeddings
            base_embeddings.append(current_layer_embeddings)
            all_embeddings.append(current_layer_embeddings)

            # 汇总所有层的嵌入，并进行求和处理以获取最终的用户和物品嵌入
        all_embeddings = torch.stack(all_embeddings[1:], dim=1)  # 从第一层开始，忽略初始嵌入
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return [base_embeddings, gnn_embeddings, hio_embeddings, fdr_embeddings]

    def _hg_inference(self):
        # 基础层的嵌入，合并用户和物品嵌入
        # base_layer_embeddings = torch.cat([self.user_embedding.weight + self.hyper_x,
        #                                    self.hyperedge_embedding.weight + self.hyper_e], dim=0)
        base_layer_embeddings = torch.cat([self.user_embedding.weight, self.hyperedge_embedding.weight], dim=0)
        check_nan(base_layer_embeddings, "Base Layer Embeddings")
        all_embeddings = [base_layer_embeddings]
        # 初始化用于存储每层嵌入的列表
        base_embeddings = []
        gnn_embeddings = []
        hio_embeddings = []
        fdr_embeddings = []  # 存储每一层的First Order Disentangled Representations

        for i in range(self.n_layers):
            # 图基础信息传递：使用图卷积更新嵌入
            gnn_layer_embeddings = torch_sparse.spmm(self.H_indices, self.H_values, self.H_shape[0],
                                                     self.H_shape[1], all_embeddings[i])
            check_nan(gnn_layer_embeddings, f"GNN Layer Embeddings {i}")

            gnn_embeddings.append(gnn_layer_embeddings)
            #高阶交互处理：用户-用户和物品-物品间的交互
            u_embeddings, e_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_h_edge], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intents, dim=1) @ self.user_intents.T
            e_int_embeddings = torch.softmax(e_embeddings @ self.hg_intents, dim=1) @ self.hg_intents.T
            hio_layer_embeddings = torch.cat([u_int_embeddings, e_int_embeddings], dim=0)

            check_nan(hio_layer_embeddings, "HIO Layer Embeddings")

            hio_embeddings.append(hio_layer_embeddings)
            # 检查和处理非法值
            has_nan = torch.isnan(gnn_layer_embeddings).any()
            has_inf = torch.isinf(gnn_layer_embeddings).any()
            if has_nan or has_inf:
                gnn_layer_embeddings[torch.isnan(gnn_layer_embeddings)] = 0  # 将 nan 值替换为 0 或其他合适的值
                gnn_layer_embeddings[torch.isinf(gnn_layer_embeddings)] = 0  # 将 inf 值替换为 0 或其他合适的值

            #First Order Disentangled Representation
            head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.hu_h_indices)
            tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.hu_t_indices)
            # edge_distributions = torch.softmax((head_embeddings * tail_embeddings) @ self.intents, dim=1) @ self.intents.T
            edge_alpha = (torch.sum(F.normalize(head_embeddings) * F.normalize(tail_embeddings), dim=1).view(
                -1) + 1) / 2
            edge_alpha = edge_alpha.flatten()

            # epsilon = 1e-12

            # edge_item_distribution = edge_distributions * edge_item_intents
            # edge_item_distribution = torch.mean(edge_item_distribution, dim=1, keepdim=False)

            # edge_distributions = torch.cat((edge_user_distribution, edge_item_distribution), dim=0)
            A_tensor = torch_sparse.SparseTensor(row=self.hu_h_indices, col=self.hu_t_indices, value=edge_alpha,
                                                 sparse_sizes=self.H_shape).cuda()
            D_scores_inv = A_tensor.sum(dim=1).nan_to_num(0, 0, 0).pow(-1).nan_to_num(0, 0, 0).view(-1)
            # edge_distributions = D_scores_inv[self.hu_h_indices] * edge_distributions

            fdr_layer_embeddings = torch_sparse.spmm(self.hu_indices, D_scores_inv[self.hu_h_indices], self.H_shape[0],
                                                     self.H_shape[1], gnn_layer_embeddings)
            fdr_layer_embeddings = torch.nn.functional.normalize(fdr_layer_embeddings, dim=1, eps=1e-8)
            fdr_embeddings.append(fdr_layer_embeddings)

            # 更新当前层的嵌入，准备传递给下一层
            current_layer_embeddings = all_embeddings[
                                           i] + gnn_layer_embeddings + hio_layer_embeddings + fdr_layer_embeddings
            base_embeddings.append(current_layer_embeddings)
            all_embeddings.append(current_layer_embeddings)

            # 汇总所有层的嵌入，并进行求和处理以获取最终的用户和物品嵌入
        all_embeddings = torch.stack(all_embeddings[1:], dim=1)  # 从第一层开始，忽略初始嵌入
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ha_embedding, self.he_embedding = torch.split(all_embeddings, [self.n_users, self.n_h_edge], 0)

        return [base_embeddings, gnn_embeddings, hio_embeddings, fdr_embeddings]


    def preprocess_embeddings(self, embeddings):
        """ Replace NaNs and Infs with zeros for stability. """
        embeddings = torch.where(torch.isnan(embeddings) | torch.isinf(embeddings), torch.zeros_like(embeddings),
                                 embeddings)
        return embeddings


    def cal_hg_ssl_loss(self, pos_users, neg_users, es):
        pos_users = torch.unique(pos_users)
        neg_users = torch.unique(neg_users)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.clamp(torch.sum(emb1 * emb2, dim=1) / self.temp, max=10))  # 避免过大的指数值
            neg_score = torch.sum(torch.exp(torch.clamp(torch.mm(emb1, emb2.T) / self.temp, max=10)), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(es[0])):
            user_embs = []
            for e in es:
                valid_e = e[i].nan_to_num(nan=0, posinf=0, neginf=0)
                u_emb, _ = torch.split(valid_e, [self.n_users, self.n_h_edge], 0)
                user_embs.append(u_emb)

            u_pos_embs = [F.normalize(x[pos_users].nan_to_num(0, 0, 0), dim=1) for x in user_embs]
            u_neg_embs = [F.normalize(x[neg_users].nan_to_num(0, 0, 0), dim=1) for x in user_embs]

            target_user_emb = u_pos_embs[0]
            for x in u_pos_embs[1:]:
                cl_loss += cal_loss(target_user_emb, x)

            target_item_emb = u_neg_embs[0]
            for x in u_neg_embs[1:]:
                cl_loss += cal_loss(target_item_emb, x)
        return cl_loss




    def cal_ssl_loss(self, users, items, es):

        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0


        def cal_loss(emb1, emb2):
            # 使用 log_softmax 来计算 log-probabilities，这是数值稳定的方式
            log_probs = torch.log_softmax(emb1 @ emb2.T / self.temp, dim=1)
            pos_probs = log_probs.diagonal()

            # 使用 logsumexp 来计算负样本概率的 log-sum-exp，这是一个数值稳定的操作
            neg_probs_logsumexp = torch.logsumexp(emb1 @ emb2.T / self.temp, dim=1)

            # 计算对比损失
            loss = -torch.mean(pos_probs - neg_probs_logsumexp)
            return loss

        for i in range(len(es[0])):
            user_embs = []
            item_embs = []
            for e in es:
                u_emb, i_emb = torch.split(e[i], [self.n_users, self.n_items], 0)
                user_embs.append(u_emb)
                item_embs.append(i_emb)

            user_embs = [F.normalize(x[users], dim=1) for x in user_embs]
            item_embs = [F.normalize(x[items], dim=1) for x in item_embs]

            target_user_emb = user_embs[0]
            for x in user_embs[1:]:
                cl_loss += cal_loss(target_user_emb, x)

            target_item_emb = item_embs[0]
            for x in item_embs[1:]:
                cl_loss += cal_loss(target_item_emb, x)

        return cl_loss



    def forward(self, users, pos_trust, neg_trust, social_hg, pos_items, neg_items):

        users, pos_trust, neg_trust = users.long().to(self.device), pos_trust.long().to(
            self.device), neg_trust.long().to(self.device)
        pos_items = pos_items.long().to(self.device)
        neg_items = neg_items.long().to(self.device)

        self.ui_embedding = self.light_layer(self.bi_graph)  # user_embs, item_embs
        check_nan(self.ui_embedding, "UI Embedding Output")
        # build hyperedge features
        self.u_embs, self.i_embs = torch.split(self.ui_embedding, [self.n_users, self.n_items], dim=0)

        # hyperedge attribute
        edge_embs = self.hyper_edge_embedding.weight
        edge_x = social_hg.smoothing_with_HGNN(edge_embs)
        edge_layer = self.edge_theta(edge_x)  # 暗含的用户信息, 直接是超边特征
        check_nan(edge_layer, "Edge Layer Output")
        self.u_embs = self.u_embs + edge_layer


        # generate the user hidden feature
        if self.conv_name in ['hgnnp', 'uingcn', 'unigin', 'unigat', 'unisage']:
            hyper_x_lats = []
            for _ in range(self.n_layers):
                _x = self.Conv_Layer(self.u_embs, social_hg)
                hyper_x_lats.append(_x)
            hyper_x_lats = sum(hyper_x_lats)
        else:
            hyper_x_lats = self.Conv_Layer(self.u_embs, social_hg)

        # or_gat_x = self.Trustor_UniGAT_Layer(self.u_embs, or_hg)
        # ee_gat_x = self.Trustee_UniGAT_Layer(self.u_embs, ee_hg)  # B * F * Emb

        # or_gat_x = torch.concat((or_gat_x, hyper_x_lats), dim=1)
        # ee_gat_x = torch.concat((ee_gat_x, hyper_x_lats), dim=1)

        # self.l_or_x = self.linear_layer(or_gat_x)
        # self.l_ee_x = self.linear_layer(ee_gat_x)
        # self.hyper_x = self.linear_layer(hyper_x_lats)
        self.hyper_x = hyper_x_lats
        self.hyper_e = social_hg.v2e(hyper_x_lats)
        embs = self.inference()
        h_embs = self._hg_inference()
        e2v_embs = social_hg.e2v(self.he_embedding)
        # concat
        #self.ha_embedding = self.ha_embedding + e2v_embs

        # user MLP
        # self.ua_embedding 来自于 user-item 解耦表征得到
        # self.ha_embedding: 来自于 user-hyperedge 解耦表征得到
        # e2v_embs: 来自于 hyperedge update user
        # 注意： 如果只选择两个用户表征进行拼接，请使用self.linear_layer
        # 例如1： user_embs = torch.cat((self.ua_embedding, self.ha_embedding), dim=1)
        # 例如2： user_embs = self.linear_layer(user_embs)
        #user_embs = torch.cat((self.ua_embedding, self.ha_embedding), dim=1)

        user_embs = torch.cat((self.ua_embedding, self.ha_embedding, e2v_embs), dim=1)
        check_nan(user_embs, "U Embeddings after adding edge layer")

        user_embs = self.mlp(user_embs)

        # bpr
        u_embeddings = user_embs[users]
        pos_embeddings = user_embs[pos_trust]
        neg_embeddings = user_embs[neg_trust]
        # output = (u_embeddings * pos_embeddings).sum(-1)
        # #output = self.out_fc(torch.concat([u_embeddings, pos_embeddings], dim=-1)).squeeze()
        # labels = labels.float()
        # cross_loss = F.binary_cross_entropy_with_logits(output, labels)

        # items embedding
        # pos_i_embeddings = self.ia_embedding[pos_items]
        # neg_i_embeddings = self.ia_embedding[neg_items]
        #
        # pos_scores1 = torch.sum(u_embeddings * pos_embeddings, 1)
        # neg_scores1 = torch.sum(u_embeddings * neg_embeddings, 1)
        #
        # # mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        # mf_loss1 = torch.mean(self.softplus(neg_scores1 - pos_scores1))


        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        pos_scores = torch.clamp(pos_scores, max=1e15)  # 避免出现过小的值
        neg_scores = torch.clamp(neg_scores, max=1e15)  # 避免出现过小的值
        mf_loss1 = -torch.log(torch.sigmoid(pos_scores) + 1e-15) - torch.log(1 - torch.sigmoid(neg_scores) + 1e-15)
        mf_loss = mf_loss1

        mf_loss = mf_loss.mean()

        # intent prototypes
        cen_loss = (self.intents.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss

        # intent distance loss
        dis_loss = torch.mean(self.intents.T @ self.intents)
        dis_loss = self.cen_dis * dis_loss

        # self-supervise learning
        # user-item ssl
        cl_loss1 = self.ssl_reg * self.cal_ssl_loss(users, pos_items, embs)
        # hypergraph ssl
        cl_loss2 = self.ssl_reg * self.cal_hg_ssl_loss(pos_trust, neg_trust, h_embs)
        cl_loss =  0.2 * cl_loss1 + cl_loss2



        #node level loss
        n_loss = self.node_level_loss(pos_embeddings, neg_embeddings, True)
        #hyperedge level loss
        he_loss = self.node_level_loss(self.hyper_e, self.he_embedding, True)
        node_loss = n_loss + 0.01*he_loss


        return mf_loss, cen_loss, dis_loss, cl_loss, node_loss

    def predict(self, trustor, trustee):
        trustor_x = self.ha_embedding[trustor]
        trustee_x = self.ha_embedding[trustee]
        # torch.sum(torch.matmul(trustor_x, trustee_x.T), dim=1)
        # torch.max(torch.matmul(trustor_x, trustee_x.T), dim=1)
        # out = self.MLP(torch.concat([trustor_x * trustee_x], dim=-1)).squeeze()
        out = torch.sum(trustor_x * trustee_x, 1)
        return out

    def compute_bce_loss(self, trustor, trustee, labels):
        trustor_x = self.ha_embedding[trustor]
        trustee_x = self.ha_embedding[trustee]
        logits = torch.sum(trustor_x * trustee_x, 1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def f(self, x, tau):
        return torch.exp(x / tau)

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float):
        between_sim = self.f(self.cosine_similarity(h1, h2), tau)
        return -torch.log(between_sim.diag() / between_sim.sum(1))

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, mean: bool):
        l1 = self.__semi_loss(z1, z2, tau)
        l2 = self.__semi_loss(z2, z1, tau)
        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    def node_level_loss(self, n1: Tensor, n2: Tensor, mean: bool = True):
        loss = self.__loss(n1, n2, self.tau, mean)
        return loss

    #正在构建
    # def membership_level_loss(self, n: Tensor, e: Tensor, hyperedge_index: Tensor, tau: float,
    #                           mean: bool = True):
    #     # 这里简单使用余弦相似度作为相似度度量
    #     def cosine_similarity(x, y):
    #         return F.cosine_similarity(x, y.unsqueeze(0))
    #
    #     # 正样本的相似度计算
    #     pos_sim = cosine_similarity(n[hyperedge_index[0]], e[hyperedge_index[1]])
    #
    #     # 生成负样本的索引
    #     num_neg_samples = hyperedge_index.shape[1]
    #     neg_indices = torch.randint(0, e.size(0), (num_neg_samples,), device=n.device)
    #
    #     # 负样本的相似度计算
    #     neg_sim = cosine_similarity(n[hyperedge_index[0]], e[neg_indices])


class LightGCN(nn.Module):

    def __init__(self, num_users: int, num_items: int, emb_dim: int, num_layers: int = 3, drop_rate: float = 0.0):
        super(LightGCN, self).__init__()
        self.num_users, self.num_items = num_users, num_items
        self.num_layers = 2
        self.drop_rate = drop_rate
        self.u_embedding = nn.Embedding(num_users, emb_dim)
        self.i_embedding = nn.Embedding(num_items, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        r"""
            Initialize learnable parameters.
        """
        nn.init.normal_(self.u_embedding.weight, 0, 0.1)  # learning all users features
        nn.init.normal_(self.i_embedding.weight, 0, 0.1)  # learning all items features

    def forward(self, ui_bigraph: BiGraph):
        r"""
        The forward function.
        Args:
            ``ui_bigraph`` (``dhg.BiGraph``): The user-item bipartite graph.
        """
        drop_rate = self.drop_rate if self.training else 0.0
        u_embs = self.u_embedding.weight
        i_embs = self.i_embedding.weight
        all_embs = torch.cat([u_embs, i_embs], dim=0)
        all_embs = all_embs.cuda()
        last_embs = all_embs
        embs_list = [all_embs]
        # Add residual connections
        for _ in range(self.num_layers):
            all_embs = ui_bigraph.smoothing_with_GCN(all_embs, drop_rate=drop_rate)
            embs_list.append(all_embs + last_embs)
            last_embs = all_embs
        embs = torch.stack(embs_list, dim=1)
        embs = torch.mean(embs, dim=1)
        # learned user and item features  Don't split
        # u_embs, i_embs = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return embs


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), discrim=False, dropout=0.6):
        super(MLP, self).__init__()
        dims = [input_dim] + list(hidden_size) + [output_dim]
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()  # 添加批量归一化层
        self.dropout = dropout

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最后一层前添加Dropout和BatchNorm
                self.dropouts.append(nn.Dropout(dropout))
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid() if discrim else None

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(self.dropouts):
                x = self.activation(x)
                x = self.batch_norms[i](x)
                x = self.dropouts[i](x)
        x = self.layers[-1](x)
        if self.sigmoid:
            x = self.sigmoid(x)
        return x

    def reset_parameters(self):
        for i in range(len(self.dim) - 1):
            nn.init.xavier_uniform_(self.layers[i].weight)
            self.layers[i].bias.data.fill_(0.0)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def normalize_adj_random_walk(adj):
    # 计算度矩阵的逆
    degree = torch.sum(adj, dim=1)
    degree_inv = torch.diag(1.0 / degree)
    adj_normalized = torch.mm(degree_inv, adj)
    return adj_normalized

def dropout_adj(adj, drop_rate, num_nodes):
    # 随机生成一个掩码来决定哪些边被丢弃
    mask = torch.rand(adj.size()) > drop_rate
    adj_dropped = adj * mask.float()
    return adj_dropped