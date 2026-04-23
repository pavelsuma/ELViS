import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .matcher import Matcher, _BASE_URL


@Matcher.register('elvis')
class Elvis(Matcher):
    def __init__(self,
                 desc_name,
                 local_dim,
                 model_dim=128,
                 embed_dim=16,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 proj_dim=64,
                 num_classes=1,
                 sh_iter=10,
                 sh_temp=10.,
                 binarized=True,
                 pretrained=None):
        super().__init__(desc_name, local_dim, model_dim, binarized)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.sh_iter = sh_iter

        self.f = nn.Sequential(
            nn.Linear(1, embed_dim, bias=True),
            norm_layer(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1, bias=True),
        )

        self.g = nn.Sequential(
            nn.Linear(1, proj_dim, bias=True),
            nn.GELU(),
            nn.Linear(proj_dim, num_classes, bias=True),
        )

        self.dust_bin = nn.Parameter(torch.tensor(1.))
        self.temp = torch.tensor(math.log(sh_temp))

        self.dust_bin_mlp = nn.Sequential(
            nn.Linear(128, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 1, bias=True),
            nn.Tanh(),
        )

        if pretrained is not None:
            self.load_state_dict(torch.hub.load_state_dict_from_url(os.path.join(_BASE_URL, 'elvis', 'networks', pretrained),
                                                                    map_location=torch.device('cpu'))['state'])
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,
                src_local=None, src_mask=None,
                tgt_local=None, tgt_mask=None,
                return_logits=False):

        src_local = self.remap_local(src_local)
        tgt_local = self.remap_local(tgt_local)

        src_local = F.normalize(src_local, p=2, dim=-1)
        tgt_local = F.normalize(tgt_local, p=2, dim=-1)

        src_dust_bin = self.dust_bin_mlp(src_local).squeeze(-1)
        tgt_dust_bin = self.dust_bin_mlp(tgt_local).squeeze(-1)

        correlation = torch.einsum('bid,bjd->bij', src_local, tgt_local)
        if src_mask is not None and tgt_mask is not None:
            corr_mask = ~src_mask[..., None] & ~tgt_mask[:, None]
            correlation = correlation.masked_fill(~corr_mask, float('-inf'))
            m = ~torch.cat([src_mask, tgt_mask], dim=-1)[..., None]
        else:
            m = torch.ones((1, src_local.shape[1] + tgt_local.shape[1], 1), device=src_local.device)

        correlation = get_matching_probs(correlation, self.dust_bin, self.sh_iter, 1 / torch.exp(self.temp), src_dust_bin, tgt_dust_bin)
        correlation = correlation.exp()

        sim_q, select_q = torch.max(correlation, dim=2, keepdim=True)
        sim_k, select_k = torch.max(correlation, dim=1, keepdim=True)
        x = torch.cat([sim_q, sim_k.permute((0, 2, 1))], dim=1)
        x = self.f(x)

        x = torch.sigmoid(x) * m
        x = x.sum(1)

        if self.training:
            x = self.g(x)

        if return_logits:
            return x.view(-1), x
        return x.view(-1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.

    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=None, num_iters=10, reg=1., src_dust_bin=None, tgt_dust_bin=None):
    """sinkhorn"""
    batch_size, m, n = S.size()
    if dustbin_score is not None:
        # augment scores matrix
        S_aug = torch.empty(batch_size, m + 1, n + 1, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        if src_dust_bin is None or tgt_dust_bin is None:
            S_aug[:, m, :] = dustbin_score
            S_aug[:, :, n] = dustbin_score
        else:
            S_aug[:, m, :-1] = tgt_dust_bin
            S_aug[:, :-1, n] = src_dust_bin
            S_aug[:, m, n] = dustbin_score

        # prepare normalized source and target log-weights
        norm = -torch.tensor(n + m, device=S.device).log()
        log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n + 1).contiguous()
        log_a[-1] += math.log(n)
        log_b[-1] += math.log(m)

        log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
        log_P = log_otp_solver(
            log_a,
            log_b,
            S_aug,
            num_iters=num_iters,
            reg=reg
        )
        return (log_P - norm)[:, :-1, :-1]
    else:
        # prepare normalized source and target log-weights
        norm = -torch.tensor(n + m, device=S.device).log()
        log_a, log_b = norm.expand(m).contiguous(), norm.expand(n).contiguous()
        log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
        log_P = log_otp_solver(
            log_a,
            log_b,
            S,
            num_iters=num_iters,
            reg=reg
        )
        return log_P - norm