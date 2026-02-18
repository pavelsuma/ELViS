import argparse
import os.path
from glob import glob

import h5py
import os.path as osp
import numpy as np
import torch

from elvis.utils.utils import pickle_save, pickle_load
from elvis.utils.revisited import compute_metrics


def load_or_combine(feat_dir, desc_name, file_type, global_type):
    file_type = '_' + file_type if file_type is not None else ''
    file_name = f'{desc_name}{file_type}_{global_type}.hdf5'

    if os.path.exists(osp.join(feat_dir, file_name)):
        with h5py.File(os.path.join(feat_dir, file_name), 'r') as f:
            desc = f['features'][:]
        print(f"Loaded {osp.join(feat_dir, file_name)}")
    else:
        splits = sorted(glob(osp.join(feat_dir, f'{desc_name}_xa?_{global_type}.hdf5')))
        if len(splits):
            images_per_hdf5 = []
            for h in splits:
                with h5py.File(os.path.join(feat_dir, h), 'r') as f:
                    images_per_hdf5.append(f['features'][:])
            desc = np.concatenate(images_per_hdf5)
            with h5py.File(os.path.join(feat_dir, file_name), "w") as f:
                f.create_dataset("features", data=desc, dtype=desc.dtype)
        else:
            raise "No splits to combine."

    return desc


def load_paths(feat_dir):
    with open(osp.join(feat_dir, 'test_query.txt')) as fid:
        query_lines = fid.read().splitlines()
    with open(osp.join(feat_dir, 'test_gallery.txt')) as fid:
        gallery_lines = fid.read().splitlines()
    return query_lines, gallery_lines

def main():
    parser = argparse.ArgumentParser(description='Compute and store image similarities and ranking with global descriptors.')
    parser.add_argument('--dataset', help='Dataset name to load descriptors of.')
    parser.add_argument('--desc_name', default='dinov2', help='Descriptors to load based on name.')
    parser.add_argument('--data_root', default='data', help='Root folder of the saved descriptor sets.')
    parser.add_argument('--shortlist_size', default=1600, type=int, help='Number of nearest neighbors of each sample.')

    args = parser.parse_args()
    dataset = args.dataset
    feat_dir = os.path.join(args.data_root, dataset)
    desc_name = args.desc_name
    m = args.shortlist_size
    global_type = 'cls' if 'dinov2' in desc_name else 'global'
    desc, gnd, query_lines, gallery_lines = None, None, None, None

    if dataset == 'gldv2':
        desc = load_or_combine(feat_dir, desc_name, None, global_type)
        query = desc

    elif dataset == 'revisitop1m' or dataset[-3:] == '+1m':
        desc = load_or_combine(osp.join(args.data_root, 'revisitop1m'), desc_name, None, global_type)
        query = desc

    if dataset.startswith(('roxford5k', 'rparis6k', 'gldv2-test')):
        small_feat_dir = os.path.join(args.data_root, dataset.split('+')[0])
        gnd = pickle_load(osp.join(feat_dir, f'gnd_{dataset}.pkl'))['gnd']

        query = load_or_combine(small_feat_dir, desc_name, 'query', global_type)
        db_desc = load_or_combine(small_feat_dir, desc_name, 'gallery', global_type)

        if desc is not None:
            desc = np.concatenate((db_desc, desc))
            with h5py.File(os.path.join(feat_dir, f'{desc_name}_gallery_{global_type}.hdf5'), "w") as f:
                f.create_dataset("features", data=desc, dtype=desc.dtype)
        else:
            desc = db_desc

    self = 0
    if dataset in ['gldv2']:
        self = 1
    elif dataset.startswith(('roxford5k', 'rparis6k')):
        m = len(desc)

    output_path = osp.join(args.data_root, dataset, f'nn_{desc_name}.pkl')

    query = torch.from_numpy(query).float()
    desc = torch.from_numpy(desc).float()
    if torch.cuda.is_available():
        query = query.cuda()
        desc = desc.cuda()
    scores = query @ desc.T
    scores, ind = torch.topk(scores, k=m + self)
    scores = torch.stack((scores, ind)).cpu()[..., self:]

    if dataset == 'gldv2' or dataset == 'sop' or dataset == 'rp2k':
        pickle_save(output_path, scores)
    else:
        pickle_save(output_path, scores)
        class Q:
            def __init__(self):
                self.name = dataset

        out, map, aps = compute_metrics(Q(), scores[1].numpy().T, gnd)


if __name__ == '__main__':
    main()