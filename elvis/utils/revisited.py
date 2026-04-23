import logging
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)


def mean_average_precision(ranks, nres, qcls, cls, kappas=[]):
    """
    Computes (mean) average precision, recall.
    Assumes each image belongs to exactly one class only

    Arguments
    ---------
    ranks : zero-based ranks of positive images QxDB
    nres  : number of positive images for each query Qx1
    qcls  : array/tensor of class ids for each query Qx1
    cls   : array/tensor of class ids for each image Dx1
    kappas: list of kappas for metric @ k

    Returns
    -------
    map    : mean average precision over all queries
    aps    : average precision per query
    apk    : average precision @ given kappas
    rec    : recall @ given kappas
    """

    apk = np.zeros(len(kappas))
    rec = np.zeros(len(kappas))

    mask = (cls[ranks] == qcls).T
    prec = np.cumsum(mask, axis=1) / (np.arange(mask.shape[1]) + 1)
    aps = (prec * mask).sum(1) / np.minimum(ranks.shape[0], nres)
    for j, k in enumerate(kappas):
        apk[j] = np.mean((prec[:, :k] * mask[:, :k]).sum(1) / np.minimum(k, nres))
        rec[j] = np.mean(np.any(mask[:, :k], axis=1))
    map = aps.mean()

    return map, aps, apk, rec

def compute_rectangular_ap(ranks, nres):
    if len(ranks) < 1:
        return 0.

    mask = np.zeros(ranks.max() + 1)
    mask[ranks] = 1
    prec = np.cumsum(mask) / (np.arange(mask.shape[0]) + 1)
    return (mask * prec).sum() / nres


def compute_trapezoidal_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zero-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[], ap_f=compute_trapezoidal_ap):
    """
    Computes the mAP for a given set of returned results.

         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only

           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    rec = np.zeros(len(kappas))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = ap_f(pos, min(len(qgnd), ranks.shape[0]))
        map = map + ap
        aps[i] = ap

        # compute recall @ k
        for j in np.arange(len(kappas)):
            ak_pos = pos[pos <= kappas[j] - 1]
            rec[j] += ak_pos.shape[0] > 0

        # compute precision @ k
        if len(pos) > 0:
            pos += 1 # get it to 1-based
            for j in np.arange(len(kappas)):
                kq = min(max(pos), kappas[j]);
                prs[i, j] = (pos <= kq).sum() / kq
            pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    rec = rec / (nq - nempty)

    return map, aps, pr, prs, rec


def compute_metrics(query_dataset, gallery_dataset, ranks, gnd, kappas=[1, 5, 10], hard=False):
    dataset_name = query_dataset.name
    ranks = ranks[:int(query_dataset.map_k)].astype(int)
    out = {}

    # old evaluation protocol
    if dataset_name.startswith(('instre', 'product1m')):
        map, aps, _, _, rec = compute_map(ranks, gnd, kappas=[1, 10, 100], ap_f=compute_rectangular_ap)
        out = {'map': map, 'aps': aps, f'r@1': rec[0], f'r@10': rec[1], f'r@100': rec[2]}

    elif dataset_name == 'gldv2-test':
        pub_map, pub_aps, _, _, _ = compute_map(ranks[:, :-750], gnd[:-750], ap_f=compute_rectangular_ap)
        map, aps, _, _, _ = compute_map(ranks[:, -750:], gnd[-750:], ap_f=compute_rectangular_ap)
        comb_map, comb_aps, _, _, _ = compute_map(ranks, gnd, ap_f=compute_rectangular_ap)
        out = {'map': map, 'aps': aps, 'pub_map': pub_map, 'comb_map': comb_map}

    elif dataset_name == 'gldv2-val':
        map, aps, _, _, _ = compute_map(ranks, gnd[:-750], ap_f=compute_rectangular_ap)
        out = {'map': map, 'aps': aps}

    elif dataset_name.startswith(('sop', 'rp2k', 'met')):
        qcls = np.asarray([int(entry.split(',')[1]) for entry in query_dataset.lines])
        cls = np.asarray([int(entry.split(',')[1]) for entry in gallery_dataset.lines])
        i, c = np.unique(cls, return_counts=True)
        i = np.searchsorted(i, qcls)
        n_pos = c[i] - 1 if dataset_name.startswith(('sop', 'rp2k')) else c[i] # hack to solve query set which is a subset of db

        map, aps, apk, rec = mean_average_precision(ranks, nres=n_pos, cls=cls, qcls=qcls, kappas=kappas)
        out = {'map': map, 'aps': aps, f'r@1': rec[0], f'r@5': rec[1], f'ap@5': apk[1]}

    elif dataset_name.startswith(('roxford', 'rparis')):

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            if hard:
                g['ok'] = np.concatenate([gnd[i]['hard']])
                g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            else:
                g['ok'] = np.concatenate([gnd[i]['hard'], gnd[i]['easy']])
                g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)

        map, aps, _, _, _ = compute_map(ranks, gnd_t, kappas)
        prefix = 'H' if hard else 'M'
        out = {f'{prefix}_map': map, f'{prefix}_aps': aps}
    else:
        log.warning(f'Evaluation protocol for {dataset_name} is not implemented yet!')

    log.info({key: f"{value * 100:.5f}" for key, value in out.items() if not key.endswith('aps')})

    return out