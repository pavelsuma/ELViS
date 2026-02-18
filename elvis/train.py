import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
import hydra

from models import matcher
from models.ames import AMES
from models.elvis import Elvis
from utils.utils import set_seed
from utils.training import train_one_epoch
from evaluate import evaluate
from utils.dataset_loader import get_test_loaders, get_train_loader

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="train")
def run(cfg: DictConfig):
    config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    score = main(cfg)
    return score

def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    set_seed(cfg.seed)
    start_epoch = 0

    student = matcher.Matcher.build_model(desc_name=cfg.desc_name, local_dim=cfg.dim_local_features, **cfg.model)

    student.to(device)
    parameters = [{'params': student.parameters()}]
    optimizer = AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.comp_fp16)

    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint['state'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint
        log.info(f'Resuming from epoch {start_epoch}.')

    teacher, beta = None, 0.
    variable_desc = cfg.desc_variable_min
    if cfg.teacher is not None:
        cfg.model.binarized = False
        teacher = matcher.Matcher.build_model(desc_name=cfg.desc_name, local_dim=cfg.dim_local_features, **cfg.model, pretrained=cfg.teacher).to(device).eval()
        beta = cfg.beta
        variable_desc = None
        log.info('Loaded the teacher model.')

    train_loader = get_train_loader(cfg.desc_name, cfg.train_dataset, variable_desc=variable_desc, num_workers=cfg.num_workers)
    val_gallery_loader, val_query_loader = None, None

    if 'val_dataset' in cfg and (cfg.val_freq or 0) > 0:
        val_query_loader, val_gallery_loader = get_test_loaders(cfg.desc_name, cfg.val_dataset, cfg.num_workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader), last_epoch=start_epoch * len(train_loader) if start_epoch > 0 else -1)

    for epoch in range(start_epoch, cfg.epochs):

        torch.cuda.empty_cache()
        train_one_epoch(student=student, teacher=teacher, loader=train_loader, variable_desc=cfg.desc_variable_min,
                       class_loss=nn.BCEWithLogitsLoss(), dist_loss=nn.MSELoss(), beta=beta,
                       scaler=scaler, optimizer=optimizer, scheduler=scheduler, epoch=epoch, freq=cfg.print_freq)

        torch.cuda.empty_cache()

        if val_query_loader and (epoch+1) % cfg.val_freq == 0:
            val_map = evaluate(model=student, lamb=cfg.val_dataset.lamb, temp=cfg.val_dataset.temp,
                               num_rerank=cfg.val_dataset.num_rerank,
                               query_loader=val_query_loader, gallery_loader=val_gallery_loader)

        torch.save({'state': student.state_dict(), 'optim': optimizer.state_dict(),
                    'epoch': epoch}, f'{cfg.desc_name}_ames{"_dist" if teacher else ""}.pt')


if __name__ == '__main__':
    run()