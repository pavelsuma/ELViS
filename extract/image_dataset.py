import os
import threading
import queue
import h5py
import numpy as np
from extract.utils import load_image_pil, quantization_factor
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.utils.data


class FeatureStorage:
    def __init__(self, save_dir, cfg_descriptors, extension, dtype_base, local_dtype, dataset_size, top_k, save_types):
        self.write_queue = queue.Queue(maxsize=50)
        self.writer = threading.Thread(target=self.h5_writer_thread)
        self.writer.start()

        self.storage = {
            s: self.init_storage(save_dir, cfg_descriptors, extension, s, dtype_base, local_dtype, top_k, dataset_size) for s in save_types
        }
        self.dtype_base = dtype_base
        self.local_dtype = local_dtype
        self.save_buffer = 1e5

    @staticmethod
    def init_storage(save_dir, cfg_descriptors, ext, save_type, dtype_base, local_dtype, top_k, dataset_size):
        file_path = os.path.join(save_dir, f"{cfg_descriptors.desc_name}_{ext}_{save_type}.hdf5")
        if save_type == "local":
            dtype = local_dtype
            if dtype in (np.float32, np.float16):
                shape = (dataset_size, top_k, cfg_descriptors.dim_local_features + 5)
            elif dtype == np.uint8:
                shape = (dataset_size, top_k, cfg_descriptors.dim_local_features // 48)
        else:
            dtype = dtype_base
            shape = (dataset_size, cfg_descriptors.dim_global_features)

        hdf5_file = h5py.File(file_path, "w")
        hdf5_file.create_dataset("features", shape=shape, dtype=dtype)
        return hdf5_file

    def save_local(self, metadata, descriptors, ids):
        if "local" not in self.storage:
            return
        storage_dtype = self.storage["local"]["features"].dtype
        if storage_dtype in (np.float32, np.float16):
            batch_data = np.concatenate((metadata, descriptors), axis=-1)
        else:
            batch_data = descriptors
        self.write_queue.put({"features": batch_data.copy(), "ids": ids.copy(), "save_type": "local"})

    def save_global(self, global_data, ids, save_type):
        if save_type not in self.storage:
            return
        self.write_queue.put({"features": global_data.astype(self.dtype_base).copy(), "ids": ids.copy(), "save_type": save_type})

    def h5_writer_thread(self):
        while True:
            try:
                data = self.write_queue.get()
                if data is None:
                    break

                save_type = data["save_type"]
                indices = data["ids"]
                features = data["features"]

                # Direct assignment works best if indices are a slice or sequential
                self.storage[save_type]["features"][indices] = features

            except Exception as e:
                print(f"Error in writer thread: {e}")
            finally:
                self.write_queue.task_done()

    def close(self):
        self.write_queue.put(None)
        self.writer.join()
        for hdf5_file in self.storage.values():
            hdf5_file.close()


class DataSet(torch.utils.data.Dataset):

    def __init__(self, name, data_path_base, scale_list, im_paths, map_k=100, imsize=None, patch_size=None,
                 gnd_data=None, train=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        assert os.path.exists(
            data_path_base), "Data path '{}' not found".format(data_path_base)
        self.data_path_base = data_path_base
        self.lines = im_paths
        self.gnd_data = gnd_data
        self.map_k = map_k
        self.name = name
        self.scale_list = np.asarray(scale_list)
        self.imsize = imsize
        self.ps = patch_size
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([transforms.Normalize(mean=self.mean, std=self.std)])
        self.train = train
        self._construct_dataset()

    def _construct_dataset(self):
        if self.train:
            if self.name == 'gldv2':
                samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in self.lines]
                self.categories = sorted(list(set([int(entry[1]) for entry in samples])))
                cat_to_label = dict(zip(self.categories, range(len(self.categories))))
                samples = [(entry[0], cat_to_label[entry[1]], entry[2], entry[3]) for entry in samples]
                self.targets = np.asarray([entry[1] for entry in samples])

            elif self.name == 'sop':
                self.targets = np.asarray([int(line.split(',')[1]) for line in self.lines])

        self.data_path = [os.path.join(self.data_path_base, i.split(',')[0]) for i in self.lines]


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, indices):
        im_list = []
        indices = indices if isinstance(indices, list) else [indices]

        for index in indices:
            img = TF.to_tensor(load_image_pil(self.data_path[index]))

            scales = self.scale_list
            if self.imsize is not None:
                scales = (scales * (self.imsize / max(img.shape)))

            if self.gnd_data is not None and 'bbx' in self.gnd_data[index]:
                bbx = self.gnd_data[index]["bbx"]
                img = img[:, int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]

            for scale in scales:
                if not self.train:
                    scale_x = quantization_factor(img.shape[2], scale, self.ps)
                    scale_y = quantization_factor(img.shape[1], scale, self.ps)
                    im = TF.resize(img, size=[round(scale_y * img.shape[-2]), round(scale_x * img.shape[-1])], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
                else:
                    im = transforms.RandomResizedCrop(round(scale * max(img.shape)), interpolation=transforms.InterpolationMode.BICUBIC)(img)
                im = self.transform(im)
                im_list.append(im)

        if self.train:
            return im_list, self.targets[indices]
        else:
            return im_list, indices
