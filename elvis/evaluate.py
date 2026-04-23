import os
import pickle
from typing import List, Tuple, Dict

import hydra
import numpy as np
import torch
import logging
from omegaconf import DictConfig, OmegaConf
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
        global_nn: torch.Tensor,
        lamb: List[int],
        temp: List[int],
        num_rerank: List[int]) -> Tuple[Dict[str, float], float]:
    model.eval()

    torch.cuda.empty_cache()
    out, sim = mean_average_precision_revisited_rerank(
        model,
        query_loader, gallery_loader, global_nn,
        lamb=lamb,
        temp=temp,
        top_k=num_rerank,
        gnd=query_loader.dataset.gnd_data,
    )

    best_score = max(out.values(), key=lambda v: v['map'])
    formatted_out = {f"{query_loader.dataset.name}/{key}": np.round(value * 100, 5) for key, value in best_score.items() if not key.endswith('aps')}
    log.info(formatted_out)

    return out, sim, best_score['map']


@hydra.main(config_path="../conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    set_seed(cfg.seed)

    query_loader, gallery_loader, global_nn = get_test_loaders(cfg.descriptors.desc_name, cfg.test_dataset, cfg.num_workers)

    model = matcher.Matcher.build_model(desc_name=cfg.descriptors.desc_name, local_dim=cfg.descriptors.dim_local_features,
                                        pretrained=cfg.model_path if not os.path.exists(cfg.model_path) else None, **cfg.model)

    if os.path.exists(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    model.eval()

    metrics, sim, best_score = evaluate(model=model, lamb=cfg.test_dataset.lamb, temp=cfg.test_dataset.temp, num_rerank=cfg.test_dataset.num_rerank,
                   query_loader=query_loader, gallery_loader=gallery_loader, global_nn=global_nn)
    out = {'metrics': metrics,
           'scores': sim,
           'config': OmegaConf.to_container(cfg, resolve=True)
           }
    with open(f'results_{cfg.descriptors.desc_name}_{cfg.test_dataset.nn_file}', 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    return map

if __name__ == '__main__':
    main()