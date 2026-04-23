import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
import hydra

from models import matcher
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

def get_scheduler(optimizer, loader_length, cfg, epoch=0, warmup_ratio=0.1):
    steps = cfg.epochs * loader_length

    if cfg.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps,
                                                               last_epoch=epoch * loader_length if epoch > 0 else -1)
    elif cfg.scheduler == 'warmcos':
        warmup_steps = int(warmup_ratio * steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps - warmup_steps)
        resume_step = epoch * loader_length if epoch > 0 else -1
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
            last_epoch=resume_step
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=steps)

    return scheduler

def main(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    set_seed(cfg.seed)
    start_epoch = 0

    student = matcher.Matcher.build_model(desc_name=cfg.descriptors.desc_name, local_dim=cfg.descriptors.dim_local_features, **cfg.model)

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
        teacher = matcher.Matcher.build_model(desc_name=cfg.descriptors.desc_name, local_dim=cfg.descriptors.dim_local_features, **cfg.model, pretrained=cfg.teacher).to(device).eval()
        beta = cfg.beta
        variable_desc = None
        log.info('Loaded the teacher model.')

    train_loader = get_train_loader(cfg.descriptors.desc_name, cfg.train_dataset, variable_desc=variable_desc, num_workers=cfg.num_workers)
    val_gallery_loader, val_query_loader = None, None

    if 'val_dataset' in cfg and (cfg.val_freq or 0) > 0:
        val_query_loader, val_gallery_loader, val_global_nn = get_test_loaders(cfg.descriptors.desc_name, cfg.val_dataset, cfg.num_workers)

    scheduler = get_scheduler(optimizer, len(train_loader), cfg, epoch=start_epoch)

    for epoch in range(start_epoch, cfg.epochs):

        torch.cuda.empty_cache()
        train_one_epoch(student=student, teacher=teacher, loader=train_loader, variable_desc=cfg.desc_variable_min,
                       class_loss=nn.BCEWithLogitsLoss(), dist_loss=nn.MSELoss(), beta=beta,
                       scaler=scaler, optimizer=optimizer, scheduler=scheduler, epoch=epoch, freq=cfg.print_freq)

        torch.cuda.empty_cache()

        if val_query_loader and (epoch+1) % cfg.val_freq == 0:
            val_map = evaluate(model=student, lamb=cfg.val_dataset.lamb, temp=cfg.val_dataset.temp,
                               num_rerank=cfg.val_dataset.num_rerank, global_nn=val_global_nn,
                               query_loader=val_query_loader, gallery_loader=val_gallery_loader)

        torch.save({'state': student.state_dict(), 'optim': optimizer.state_dict(),
                    'epoch': epoch}, f'{cfg.descriptors.desc_name}_elvis{"_dist" if teacher else ""}.pt')


if __name__ == '__main__':
    run()