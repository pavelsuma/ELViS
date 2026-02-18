import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from extract.extract import get_local, extract_features
from .spatial_attention_2d import SpatialAttention2d

from elvis.utils.utils import TripletSampler
from elvis.utils.metrics import AverageMeter
from elvis.utils.revisited import compute_metrics
from elvis.utils.utils import set_seed, get_loss, get_optimizer, get_scheduler


def extract_global_feat(im, model, detector, chunk=15):
    im = torch.cat(im, dim=0)
    im = im.split(chunk)
    feats, cls = [], []

    for i in im:
        with torch.no_grad():
            f, c = model(i.cuda())[-1]
        feats.append(f)
        cls.append(c)

    feats = torch.cat(feats, dim=0)
    cls = torch.cat(cls)

    if detector:
        feats, weights = detector(feats)
        feats = feats * weights

    feats = F.avg_pool2d(feats, (feats.size(-2), feats.size(-1))).squeeze()
    feats = F.normalize(feats, p=2, dim=-1).unsqueeze(-2)
    return feats


def test(cfg, model, detector, batch_size, test_loaders, gnd):
    if detector:
        detector.eval()
    model.eval()

    class FeatureStorage:
        def __init__(self, dataset):
            self.storage = {'w-avg': np.zeros((len(dataset), cfg.local_desc.dim_global_features))}

        def save_global(self, feats, image_ids, save_type):
            if save_type == 'w-avg':
                self.storage[save_type][image_ids] = feats

    q = FeatureStorage(test_loaders[0].dataset)
    extract_features(model, detector, None, test_loaders[0], q, batch_size, True, topk=cfg.imsize**2)
    if self := (test_loaders[0] == test_loaders[1]):
        db = q
    else:
        db = FeatureStorage(test_loaders[1].dataset)
        extract_features(model, detector, None, test_loaders[1], db, batch_size, True, topk=cfg.imsize**2)

    print("perform global retrieval")
    # q /= np.linalg.norm(q, ord=2, axis=1, keepdims=True)
    sim = np.dot(db.storage['w-avg'], q.storage['w-avg'].T)
    ranks = np.argsort(-sim, axis=0)

    # revisited evaluation
    out = compute_metrics(test_loaders[0].dataset,
                                   test_loaders[1].dataset,
                                   ranks[self:],
                                   gnd,
                                   [1, 5, 10])
    formatted_out = {f"{test_loaders[0].dataset.name}/{key}": np.round(value * 100, 5) for key, value in out.items() if not key.endswith('aps')}
    if wandb.run is not None:
        wandb.log(formatted_out)


def get_loader(dataset, batch_size=512, num_workers=8, sampler=None, collate=None):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate
    )

def get_loaders(cfg, scale_list, mean, std, collate):
    im_paths = np.asarray(load_gnd(os.path.join(cfg.train_dataset.desc_dir, cfg.train_dataset.train_txt)))
    train_dataset = DataSet(cfg.train_dataset.name, cfg.train_dataset.data_dir, [1],
                            im_paths=im_paths, imsize=cfg.imsize, train=True, mean=mean, std=std)

    ps = cfg.patch_size
    gnd_path = os.path.join(cfg.test_dataset.desc_dir, cfg.test_dataset.test_gnd_file) if cfg.test_dataset.test_gnd_file is not None else None
    gnd_data = load_gnd(os.path.join(cfg.test_dataset.desc_dir, gnd_path)) if gnd_path else None
    im_paths = np.asarray(load_gnd(os.path.join(cfg.test_dataset.desc_dir, cfg.test_dataset.query_paths)))

    query_dataset = DataSet(cfg.test_dataset.name, cfg.test_dataset.data_dir, scale_list,
                            im_paths=im_paths, gnd_data=gnd_data, imsize=cfg.imsize, map_k=cfg.test_dataset.map_k, mean=mean, std=std,
                            patch_size=ps, train=False)
    query_loader = get_loader(query_dataset, batch_size=1, num_workers=cfg.num_workers, collate=collate)

    if cfg.test_dataset.query_paths == cfg.test_dataset.gallery_paths:
        gallery_loader = query_loader
    else:
        im_paths = np.asarray(load_gnd(os.path.join(cfg.test_dataset.desc_dir, cfg.test_dataset.gallery_paths)))
        gallery_dataset = DataSet(cfg.test_dataset.name, cfg.test_dataset.data_dir, scale_list,
                                  im_paths=im_paths, imsize=cfg.imsize,
                                  patch_size=ps, train=False, mean=mean, std=std)
        gallery_loader = get_loader(gallery_dataset, batch_size=1, num_workers=cfg.num_workers, collate=collate)

    with open(os.path.join(cfg.train_dataset.desc_dir, cfg.train_dataset.global_nn), 'rb') as fid:
        nn_cache = pickle.load(fid)
    train_sampler = TripletSampler(train_dataset.targets, 1, nn_cache,
                                   cfg.train_dataset.neg_num, cfg.train_dataset.num_candidates,
                                   cfg.train_dataset.min_pos, cfg.train_dataset.min_neg,
                                   cfg.epoch_size)
    train_loader = get_loader(train_dataset, cfg.train_dataset.batch_size, cfg.num_workers, train_sampler, collate=collate)

    return train_loader, (query_loader, gallery_loader), gnd_data


def train(model, detector, train_loader, optimizer, scheduler, loss_fn, epoch, freq=100):
    detector.train()
    device = next(detector.parameters()).device
    loader_length = len(train_loader)
    train_losses = AverageMeter(device=device, length=loader_length)
    pbar = tqdm(train_loader, ncols=80, desc='Training   [{:03d}]'.format(epoch))

    for i, entry in enumerate(pbar):
        im, targets = entry
        bsize = im[0][0].size(0)
        feats = extract_global_feat(im[0], model, detector)
        loss = loss_fn(feats[:bsize], feats[bsize:-bsize], feats[-bsize:])

        train_losses.append(loss)
        loss.backward()
        ##############################################
        optimizer.step()
        optimizer.zero_grad()

        if scheduler[-1]:
            scheduler[0].step()

        if not (i + 1) % freq:
            step = epoch + i / loader_length
            print('step/loss/lr:', step, train_losses.last_avg.item(), scheduler[0].get_last_lr()[0])
            wandb.log({'train/loss': train_losses.last_avg.item()})

        if not scheduler[-1]:
            scheduler[0].step()


@hydra.main(config_path="../conf", config_name="experiment")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    if torch.cuda.is_available() and not cfg.cpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    ############################## Loading models ##############################
    extract_f, model, mean, std, collate, _ = load_model(cfg.local_desc.desc_name, cfg.resume)
    model.to(device)
    model.eval()

    detector = SpatialAttention2d(cfg.local_desc.dim_global_features)
    detector.to(device)

    ############################## Experiment logging ##############################
    config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if cfg.wandb:
        wandb.init(project=cfg.wandb.project,
                   entity=cfg.wandb.entity,
                   config=config,
                   group=cfg.exp_name,
                   reinit=True)

    ############################## Train dataset + loader ##############################
    train_loader, test_loaders, gnd = get_loaders(cfg, [1], mean=mean, std=std, collate=collate)
    ############################## Optim, scheduler, loss ##############################
    loss = get_loss(cfg.loss, cfg)
    start_epoch = 0
    optimizer = get_optimizer([{'params': detector.parameters(), 'lr': cfg.lr}], cfg)
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint
        print('Resuming from epoch %d.'%start_epoch)
    scheduler = get_scheduler(optimizer, len(train_loader), cfg, epoch=start_epoch)

    ############################## Train loop ##############################
    for epoch in range(start_epoch, cfg.epochs):
        set_seed(cfg.seed + epoch)
        train(model, detector, train_loader, optimizer, scheduler, loss, epoch)
        if epoch % cfg.test_freq == 0:
            test(cfg, model, detector, cfg.test_dataset.batch_size, test_loaders, gnd)
        torch.save({'state': deepcopy(detector.state_dict()), 'optim': optimizer.state_dict(), 'epoch': epoch}, f'checkpoint_{epoch}.pth')

    wandb.finish()

if __name__ == "__main__":
    main()
