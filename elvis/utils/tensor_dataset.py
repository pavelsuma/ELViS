import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import pickle_load


class TensorFileDataset(Dataset):
    def __init__(self,
            name: str,
            desc_dir: str,
            local_desc_name: str,
            gnd_data=None,
    ):
        self.name = name
        self.desc_dir = desc_dir
        self.gnd_data = gnd_data
        self.local_file = os.path.join(desc_dir, local_desc_name)
        self.local_storage = h5py.File(self.local_file, 'r', rdcc_nbytes=618400 * 1e2)
        self.local_dtype = self.local_storage['features'][0].dtype
        self.num_samples = len(self.local_storage['features'])

    def __len__(self):
        return self.num_samples

    def __del__(self):
        self.local_storage.close()

class TrainDataset(TensorFileDataset):
    def __init__(self, *args, samples, db_desc_num, query_desc_num, variable_desc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_desc_num = db_desc_num
        self.query_desc_num = query_desc_num
        self.variable_desc = variable_desc

        if self.name == 'gldv2':
            samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for
                       line in samples]
            self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
            self.cat_to_label = dict(zip(self.categories, range(len(self.categories))))
            self.samples = [(entry[0], self.cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
            self.targets = [entry[1] for entry in self.samples]

        self.image_sizes = torch.Tensor([[[int(entry[-1]), int(entry[-2])]] for entry in samples])


    def __getitem__(self, batch_index):
        if self.variable_desc is not None:
            ql = np.random.randint(self.variable_desc, self.query_desc_num)
            gl = np.random.randint(self.variable_desc, self.db_desc_num)
        else:
            ql = self.query_desc_num
            gl = self.db_desc_num

        l = max(ql, gl)
        idx = np.sort(np.unique(batch_index)).tolist()

        if self.local_dtype in (np.float32, np.float16):
            all_local = self.local_storage['features'][idx, :l]
            all_local = all_local[[idx.index(i) for i in batch_index]]
            local_feat = all_local[..., 5:]
            metadata = all_local[..., :5]
        else:
            all_local = self.local_storage['features'][idx]
            all_local = all_local[[idx.index(i) for i in batch_index]]
            local_feat = all_local['descriptor'][:, :l]
            metadata = all_local['metadata'][:, :l]

        local_feat = torch.from_numpy(local_feat.reshape(-1, 3, l, local_feat.shape[-1])).float()
        metadata = metadata.reshape(-1, 3, l, metadata.shape[-1])
        masks = torch.from_numpy(metadata[..., 3]).view(-1, 3, l).bool()

        anchors_local, anchors_masks = local_feat[:, 0], masks[:, 0]
        positive_local, positive_masks = local_feat[:, 1], masks[:, 1]
        negatives_local, negatives_masks = local_feat[:, 2], masks[:, 2]

        anchors = (anchors_local[:, :ql], anchors_masks[:, :ql])
        positives = (positive_local[:, :gl], positive_masks[:, :gl])
        negatives = (negatives_local[:, :gl], negatives_masks[:, :gl].bool())

        return anchors, positives, negatives


class TestDataset(TensorFileDataset):
    def __init__(self, *args, desc_num, nn_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.desc_num = desc_num

        if nn_file is not None:
            nn_inds_path = os.path.join(self.desc_dir, nn_file)
            self.cache_nn = pickle_load(nn_inds_path)

            if self.name == 'gldv2-val':
                self.num_samples = self.num_samples - 750
                self.cache_nn = self.cache_nn[:, :-750]

    def __getitem__(self, batch_index):
        idx = np.sort(np.unique(batch_index)).tolist()
        all_local = self.local_storage['features'][idx]

        all_local = all_local[[idx.index(i) for i in batch_index]]

        if all_local.dtype in (np.float32, np.float16):
            local_feat = all_local[:, :self.desc_num, 5:]
            metadata = all_local[:, :self.desc_num, :5]
        else:
            local_feat = all_local['descriptor'][:, :self.desc_num]
            metadata = all_local['metadata'][:, :self.desc_num]

        if local_feat.dtype == np.uint8:
            local_feat = np.unpackbits(local_feat, axis=-1).astype(float)
            local_feat = 2 * local_feat - 1

        local_feat = torch.from_numpy(local_feat).float()
        masks = torch.from_numpy(metadata[..., 3]).bool()

        return (local_feat, masks), batch_index