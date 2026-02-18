import os
from functools import partial

import hydra
import numpy as np
import timm
import torch
from omegaconf import DictConfig

from extract.extract import extract
from extract.image_dataset import DataSet, FeatureStorage
from extract.spatial_attention_2d import SpatialAttention2d
from elvis.models.binarization_layer import BinarizationLayer
from elvis.utils.utils import pickle_save, pickle_load

from extract.utils import collate_batches

_BASE_URL = 'http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks/'

def load_dino(name):
    if 'dinov2' in name:
        model = torch.hub.load('facebookresearch/dinov2', name)
    else:
        model = torch.hub.load('/path/to/dinov3', name, source='local', weights='XXX')
    model.forward = partial(model.get_intermediate_layers, return_class_token=True, reshape=True)
    return model

def load_siglip(name):
    model = timm.create_model(
        name, pretrained=True, dynamic_img_size=True
    )

    def siglip_forward_hook(module, x, res):
        cls = module.pool(res[0])
        return [(res[1][-1], cls)]

    model.forward = partial(model.forward_intermediates, indices=1, norm=True)
    model.register_forward_hook(siglip_forward_hook)
    return model, model.default_cfg['mean'], model.default_cfg['std']


def load_model(model_name):
    """
    Loads the backbone model and returns components needed for extraction.
    """
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Load backbone model
    if 'dino' in model_name:
        model = load_dino(model_name)
    elif 'siglip' in model_name:
        model, mean, std = load_siglip(model_name)
    else:
        raise ValueError(f"Backbone {model_name} not supported")

    return model, mean, std


def find_unique_shortlist(im_paths, shortlist, cfg):
    nn_file = f'nn_{cfg.descriptors.desc_name}_cls.pkl'
    global_nn = pickle_load(os.path.join(cfg.dataset.desc_dir, nn_file))
    ranks = global_nn[1].long()
    unq, idx = np.unique(ranks[:, :shortlist].T, return_index=True)
    unq = unq[np.argsort(idx)]
    map_unq = dict(zip(unq, range(len(unq))))
    func = lambda x: map_unq[x] if x in map_unq else x
    mapped_ranks = np.vectorize(func)(ranks)
    global_nn = torch.cat((global_nn[:2], torch.from_numpy(mapped_ranks)[None]), dim=0)
    pickle_save(os.path.join(cfg.dataset.desc_dir, nn_file), global_nn)
    return im_paths[unq]


@hydra.main(config_path="../conf", config_name="extract")
def main(cfg: DictConfig):
    if torch.cuda.is_available() and not cfg.cpu:
        torch.backends.cuda.matmul.allow_tf32 = True

    model, mean, std = load_model(cfg.descriptors.model_name)
    model.cuda()
    model.eval()
    detector, binarizer = None, None

    # Load trained feature detector
    if cfg.detector:
        detector = SpatialAttention2d(cfg.descriptors.dim_local_features)
        detector.cuda()
        detector.eval()
        cpt = torch.hub.load_state_dict_from_url(f'{_BASE_URL}/{cfg.descriptors.desc_name}_detector.pt')
        detector.load_state_dict(cpt['state'], strict=True)

    local_dtype = np.float16 if cfg.store_fp16 else np.float32
    # Load pre-trained feature binarization module or ITQ weights
    if cfg.binarization:
        binarizer = BinarizationLayer(dims=768, bits=128, trainable=False, pretrained=False)
        cpt = torch.hub.load_state_dict_from_url(f'{_BASE_URL}/{cfg.descriptors.desc_name}_binarization.pt')
        binarizer.load_state_dict(cpt['state'], strict=True)
        binarizer.cuda().eval()
        local_dtype = np.uint8
    global_dtype = np.float16 if cfg.store_fp16 else np.float32

    if "test_gnd_file" in cfg.dataset:
        files_to_extract = ((cfg.dataset.query_paths, cfg.dataset.test_gnd_file if cfg.bbox else None, None),
                            (cfg.dataset.gallery_paths, None, cfg.shortlist))
    else:
        files_to_extract = ((cfg.dataset.train_txt, None, None),)

    for file_name, gnd_path, shortlist in files_to_extract:
        if file_name is None:
            continue

        gnd_data = load_gnd(os.path.join(cfg.dataset.desc_dir, gnd_path)) if gnd_path else None
        im_paths = np.asarray(load_gnd(os.path.join(cfg.dataset.desc_dir, file_name)))
        if shortlist is not None and shortlist > 0:
            im_paths = find_unique_shortlist(im_paths, shortlist, cfg)
        dataset = DataSet(cfg.dataset.name, cfg.dataset.data_dir, cfg.scale_list, im_paths=im_paths,
                          gnd_data=gnd_data, imsize=cfg.imsize, patch_size=cfg.descriptors.patch_size, mean=mean, std=std)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_batches
        )

        extension = file_name.split(".")[0]

        feature_storage = FeatureStorage(cfg.dataset.desc_dir, cfg.descriptors, extension, global_dtype, local_dtype,
                                         len(dataset), cfg.topk, cfg.desc_type)
        extract(model, detector, binarizer, feature_storage, dataloader, cfg.topk, cfg.batch_size, cfg.comp_fp16)
        feature_storage.close()


if __name__ == "__main__":
    main()
