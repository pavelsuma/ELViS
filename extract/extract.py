import gc
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from extract.utils import batching

log = logging.getLogger(__name__)


def fix_query_pos(feature_storage, gnd, im_paths):
    im_sizes = np.asarray([i.split(',')[-2:] for i in im_paths]).astype(int)
    loc = feature_storage.storage['local'][..., :2]
    crops = np.asarray([i['bbx'] for i in gnd])[:, None]
    loc = loc * (crops[..., 2:] - crops[..., :2])
    loc = (loc + crops[..., :2]) / im_sizes[:, None]
    feature_storage.storage['local'][..., :2] = loc

def find_divisors(number):
    divisors = np.arange(1, int(np.sqrt(number)) + 1)
    divisors = divisors[number % divisors == 0]
    return divisors


def calculate_receptive_boxes(imsize):
    pc_x, pc_y = 1 / torch.tensor(imsize)
    loc_x = torch.arange(pc_x / 2, 1., pc_x)
    loc_y = torch.arange(pc_y / 2, 1., pc_y)
    boxes = torch.stack([loc_x.repeat_interleave(loc_y.shape[0]), loc_y.tile(loc_x.shape[0])], dim=-1)
    return boxes


def get_local(local_features, local_weights, topk=100):
    features, weights, boxes, scale_limits = [], [], [], []
    last_limit = 0

    for feat, weight in zip(local_features, local_weights):
        rf_boxes = calculate_receptive_boxes(feat.shape[-2:])[None].repeat(feat.shape[0], 1, 1)
        boxes.append(rf_boxes)
        feat = feat.flatten(start_dim=-2).permute(0, 2, 1)
        features.append(feat)
        weights.append(weight.flatten(start_dim=1))
        last_limit += feat.shape[1]
        scale_limits.append(last_limit)

    feats, weights = torch.cat(features, dim=1), torch.cat(weights, dim=1)
    boxes = torch.cat(boxes, dim=1).to(feats.device)
    seq_len = min(feats.shape[1], topk)

    top_weights, indices = torch.topk(weights, k=seq_len, dim=1)
    top_feats = feats.gather(1, indices[..., None].expand(-1, -1, feats.size(-1)))
    scale_enc = torch.bucketize(indices, torch.tensor(scale_limits, device=feats.device), right=True)
    locations = boxes.gather(1, indices[..., None].expand(-1, -1, 2))

    return top_feats, top_weights, scale_enc, locations, seq_len


def process_image_batches(images, model, detector, batch_size):
    torch.cuda.empty_cache()
    gc.collect()

    batch_features = {"local": [], "local_weights": [], "cls": [], "og-avg": [], "w-avg": []}
    for batch in batching(images, batch_size):
        feats, cls = model(batch.cuda())[-1]
        cls = F.normalize(cls, p=2, dim=-1, eps=1e-15)
        if detector is not None:
            w_feats, weights = detector(feats)
        else:
            w_feats = feats
            feats = F.normalize(feats, p=2, dim=1, eps=1e-15)
            weights = torch.einsum('qd,qdji->qji', cls, feats).unsqueeze(1)

        batch_features["cls"].append(cls)
        batch_features["w-avg"].append(F.normalize((weights * w_feats).mean((2, 3)), p=2, dim=-1))
        batch_features["og-avg"].append(F.normalize((weights * feats).mean((2, 3)), p=2, dim=-1))
        feats = F.normalize(feats, p=2, dim=1, eps=1e-15)
        batch_features["local"].append(feats)
        batch_features["local_weights"].append(weights)
    return {k: torch.cat(v, dim=0) for k, v in batch_features.items()}


def store_features(features, image_ids, storage, projection, topk):
    if 'local' in storage.storage:
        local_feats, weights, scale_enc, locations, seq_len = get_local(
            [f["local"] for f in features], [f["local_weights"] for f in features], topk
        )

        local_info = np.zeros((local_feats.size(0), topk, 5))
        local_info[:, :seq_len] = torch.cat(
            [locations, scale_enc.unsqueeze(-1), torch.zeros_like(weights).unsqueeze(-1), weights.unsqueeze(-1)], dim=-1
        ).cpu().numpy()
        local_info[:, seq_len:, 3] = 1

        if projection:
            local_feats = projection(local_feats)

        local_feats = local_feats.cpu().numpy()
        if storage.local_dtype == np.uint8:
            local_feats = np.packbits(local_feats > 0, axis=-1)

        padded_feats = np.zeros((local_feats.shape[0], topk, local_feats.shape[-1]), dtype=local_feats.dtype)
        padded_feats[:, :seq_len] = local_feats

        storage.save_local(local_info, padded_feats, image_ids)

    for key in [k for k in storage.storage.keys() if k != 'local']:
        global_features = torch.stack([f[key] for f in features], dim=1).mean(dim=1)
        storage.save_global(F.normalize(global_features, p=2, dim=-1).cpu().numpy(), image_ids, key)


@torch.inference_mode()
def extract_features(model, detector, projection, loader, storage, batch_size, comp_fp16, topk=700):
    pbar = tqdm(loader, file=open(os.devnull, "w"))
    for batches in pbar:
        for image_list, image_ids in zip(*batches):
            with torch.amp.autocast("cuda", enabled=comp_fp16):
                features = [process_image_batches(img, model, detector, batch_size) for img in image_list]
            store_features(features, image_ids, storage, projection, topk)
        log.info(str(pbar))


def extract(model, detector, projection, feature_storage, dataloader, topk, batch_size, comp_fp16):
    extract_features(model, detector, projection, dataloader, feature_storage, batch_size, comp_fp16, topk=topk)
    return feature_storage.storage