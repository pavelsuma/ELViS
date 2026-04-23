import os

import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from .tensor_dataset import TestDataset, TrainDataset
from .utils import pickle_load


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


def basic_collate(batch):
    return batch[0]


class TripletSampler:
    def __init__(self, labels, batch_size, global_nn, num_candidates):
        self.batch_size     = batch_size
        self.num_candidates = num_candidates
        self.cache_nn_inds = global_nn[1].astype(np.longlong)[:, :num_candidates]
        self.cache_nn_sims = global_nn[0][:, :num_candidates]
        self.labels = np.asarray(labels)

        #############################################################################
        ## Collect valid tuples
        valids = np.zeros_like(labels)
        for i in range(len(self.cache_nn_inds)):
            nnids = self.cache_nn_inds[i]
            nsims = self.cache_nn_sims[i]
            query_label = labels[i]
            index_labels = np.array([labels[j] for j in nnids])
            if (nsims[index_labels == query_label] > 0).sum() < 5:
                continue
            if (nsims[index_labels != query_label] > 0).sum() < 5:
                continue
            valids[i] = 1
        self.valids = np.where(valids > 0)[0]
        self.num_samples = len(self.valids)

    def __iter__(self):
        batch = []
        cands = torch.randperm(self.num_samples).tolist()
        for i in range(len(cands)):
            anchor_idx = self.valids[cands[i]]
            anchor_label = self.labels[anchor_idx]

            nsims = self.cache_nn_sims[anchor_idx]
            mask = nsims > 0
            nnids = self.cache_nn_inds[anchor_idx][mask]
            nsims = nsims[mask]

            l = self.labels[nnids]

            positive_inds = np.where(l == anchor_label)[0]
            negative_inds = np.setdiff1d(np.arange(len(l)), positive_inds)
            assert(len(positive_inds) > 0)
            assert(len(negative_inds) > 0)

            pos_sims = np.power(nsims[positive_inds], 3)
            neg_sims = np.power(nsims[negative_inds], 3)
            pos = np.random.choice(positive_inds, 1, p=pos_sims/np.linalg.norm(pos_sims, ord=1))
            neg = np.random.choice(negative_inds, 1, p=neg_sims/np.linalg.norm(neg_sims, ord=1))

            batch.append(anchor_idx.item())
            batch.append(nnids[pos].item())
            batch.extend(nnids[neg].tolist())

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (self.num_samples * 3 + self.batch_size - 1) // self.batch_size


def get_train_set(desc_name, train_dataset, variable_desc):
    train_lines = read_file(os.path.join(train_dataset.desc_dir, train_dataset.train_txt))

    train_set = TrainDataset(train_dataset.name, train_dataset.desc_dir,
                                  local_desc_name=desc_name + '_local.hdf5',
                                  samples=train_lines,
                                  db_desc_num=train_dataset.db_desc_num,
                                  query_desc_num=train_dataset.query_desc_num,
                                  variable_desc=variable_desc)

    return train_set


def get_train_loader(desc_name, train_dataset, variable_desc, num_workers=8):
    global_nn = os.path.join(train_dataset.desc_dir, train_dataset.nn_file)
    global_nn = pickle_load(global_nn)

    train_set = get_train_set(desc_name, train_dataset, variable_desc)
    train_sampler = TripletSampler(train_set.targets, train_dataset.batch_size, global_nn, train_dataset.num_candidates)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=1, num_workers=num_workers, pin_memory=train_dataset.pin_memory, collate_fn=basic_collate)

    return train_loader


def get_test_sets(desc_name, test_dataset):
    test_gnd_data = None if test_dataset.test_gnd_file is None else pickle_load(os.path.join(test_dataset.desc_dir, test_dataset.test_gnd_file))['gnd']

    gallery_set = TestDataset(test_dataset.name, test_dataset.desc_dir, desc_name + '_gallery_local.hdf5',
                                   desc_num=test_dataset.db_desc_num)
    query_set   = TestDataset(test_dataset.name, test_dataset.desc_dir, desc_name + '_query_local.hdf5',
                                    desc_num=test_dataset.query_desc_num, gnd_data=test_gnd_data,
                                    map_k=test_dataset.map_k)

    return query_set, gallery_set


def get_test_loaders(desc_name, test_dataset, num_workers=8):
    query_set, gallery_set = get_test_sets(desc_name, test_dataset)

    query_sampler = BatchSampler(SequentialSampler(query_set), batch_size=test_dataset.batch_size, drop_last=False)
    gallery_sampler = BatchSampler(SequentialSampler(gallery_set), batch_size=test_dataset.batch_size, drop_last=False)

    query_loader = DataLoader(query_set, sampler=query_sampler, batch_size=1, num_workers=num_workers, pin_memory=test_dataset.pin_memory, collate_fn=basic_collate)
    gallery_loader = DataLoader(gallery_set, sampler=gallery_sampler, batch_size=1, num_workers=num_workers, pin_memory=test_dataset.pin_memory, collate_fn=basic_collate)

    global_nn = pickle_load(os.path.join(test_dataset.desc_dir, test_dataset.nn_file))

    return query_loader, gallery_loader, global_nn