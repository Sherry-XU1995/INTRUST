# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np

import torch
from dhg.datapipe import load_from_pickle, load_from_txt


class TripletDataset:
    def __init__(self, pairs):
        self.pos_pairs = [_ for _ in pairs if _[2] == 1]
        self.neg_pairs = [_ for _ in pairs if _[2] == 0]
        self.trustor_neg_map = {}
        for pair in self.neg_pairs:
            self.trustor_neg_map.setdefault(pair[0], []).append(pair[1])

    def __getitem__(self, idx):
        pos_pair = self.pos_pairs[idx]
        return pos_pair[0], pos_pair[1], random.choice(self.trustor_neg_map[pos_pair[0]])

    def __len__(self):
        return len(self.pos_pairs)


class Collator:
    def __init__(self,  data):
        """
        Don't worker
        :param data:
        """

        self.pos_pairs = [_ for _ in data if _[2] == 1]
        self.neg_pairs = [_ for _ in data if _[2] == 0]
        self.trustor_neg_map = {}
        for pair in self.neg_pairs:
            self.trustor_neg_map.setdefault(pair[0], []).append(pair[1])

    def __call__(self, data):
        trustor = []
        pos_trustee = []
        neg_trustee = []

        for pair in data:
            trustor.append(pair[0])
            pos_trustee.append(pair[1])
            neg_trustee.append(random.choice(self.trustor_neg_map[pair[0]]))


        trustor = torch.IntTensor(trustor)
        pos_trustee = torch.IntTensor(pos_trustee)
        neg_trustee = torch.IntTensor(neg_trustee)
        return trustor, pos_trustee, neg_trustee




class DataReader:
    def __init__(self, data_root: Optional[str] = None) -> None:
        self.data_root = data_root
        self.adj_list = load_from_txt(Path(data_root, 'adj_list.txt'), dtype="int", sep=" ")        # user and item bigraph
        self.rating = load_from_pickle(Path(data_root, 'rating.pkl'))       # user and item ratting
        self.train_mask = load_from_pickle(Path(data_root, 'train_mask.pkl'))   # user trustor trustee for train
        self.test_mask = load_from_pickle(Path(data_root, 'test_mask.pkl'))     # user trustor trustee for test
        self.total_trust_pair = load_from_pickle(Path(data_root, 'total_trust_pair.pkl'))   # all train pair
        self.full_adj = load_from_pickle(Path(data_root, 'full_adj.pkl'))       # all user
        self.social_adj = load_from_pickle(Path(data_root, 'social_adj.pkl'))   # user for adj
        self.u_v_info = load_from_pickle(Path(data_root, 'u_v_info.pkl'))       # user for item
        self.u_r_info = load_from_pickle(Path(data_root, 'u_r_info.pkl'))       # user and item level
        self.v_u_info = load_from_pickle(Path(data_root, 'v_u_info.pkl'))       # item for user
        self.v_r_info = load_from_pickle(Path(data_root, 'v_r_info.pkl'))       # item and user level
        self.pr_weights = load_from_pickle(Path(data_root, 'pr_weights.pkl'))   # pagerank weights

        self.n_users = len(self.u_v_info)
        self.n_items = len(self.v_u_info)

    @property
    def num_users(self):
        return len(self.u_v_info)

    @property
    def num_items(self):
        return len(self.v_u_info)

    @property
    def num_edges(self):
        return len(self.rating)

    @property
    def train_pairwise_dataset(self):
        return self.train_mask

    @property
    def test_pairwise_dataset(self):
        return self.test_mask

    @property
    def train_triplet_dataset(self):
        return TripletDataset(self.train_mask)

    @property
    def test_triplet_dataset(self):
        return TripletDataset(self.test_mask)


if __name__ == '__main__':
    ciao_reader = DataReader('./data/ciao')
    assert ciao_reader.num_users, 7317
    assert ciao_reader.num_items, 104975
    assert ciao_reader.num_edges, 283320

