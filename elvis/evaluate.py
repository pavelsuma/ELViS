import os
from typing import List, Tuple, Dict

import hydra
import torch
import logging
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from models import matcher
from models.ames import AMES
from models.elvis import Elvis
from utils.metrics import mean_average_precision_revisited_rerank
from utils.utils import set_seed
from utils.dataset_loader import get_test_loaders

log = logging.getLogger(__name__)

@torch.inference_mode()
def evaluate(
        model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        lamb: List[int],
        temp: List[int],
        num_rerank: List[int]) -> Tuple[Dict[str, float], float]:
    model.eval()

    torch.cuda.empty_cache()
    metrics = mean_average_precision_revisited_rerank(
        model,
        query_loader, gallery_loader, query_loader.dataset.cache_nn,
        lamb=lamb,
        temp=temp,
        top_k=num_rerank,
        gnd=query_loader.dataset.gnd_data,
    )
    return metrics


@hydra.main(config_path="../conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    set_seed(cfg.seed)

    query_loader, gallery_loader = get_test_loaders(cfg.desc_name, cfg.test_dataset, cfg.num_workers)

    model = matcher.Matcher.build_model(desc_name=cfg.desc_name, local_dim=cfg.dim_local_features,
                                        pretrained=cfg.model_path if not os.path.exists(cfg.model_path) else None, **cfg.model)

    if os.path.exists(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    model.eval()

    map = evaluate(model=model, lamb=cfg.test_dataset.lamb, temp=cfg.test_dataset.temp, num_rerank=cfg.test_dataset.num_rerank,
                   query_loader=query_loader, gallery_loader=gallery_loader)
    return map

if __name__ == '__main__':
    main()