# ELViS: Efficient Visual Similarity From Local Descriptors That Generalizes Across Domains
***

This repository contains the code for the paper ["ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains"](https://arxiv.org/abs/2603.28603), by the authors Pavel Suma, Giorgos Kordopatis-Zilos, Yannis Kalantidis, and Girogos Tolias.
In Proceedings of ICLR 2026

## TLDR

ELViS is a fast, lightweight and interpretable module for estimating image-to-image similarity. It leverages pre-trained foundation models, such as DINOv2, DINOv3, or SigLIP2 and generalizes well to many image domains.


***

## Setup

This code was implemented using Python 3.12.3 and the following dependencies:

```
torch==2.7.0
hydra-core==1.3.2
numpy==2.1.2
tqdm==4.66.5
h5py==3.12.1
```

You can install them via pip:
```
pip install -r requirements.txt
```

***

## Quickstart

We provide ELViS trained on GLDv2 in three variants. DINOv2 or CVNet backbone: `dinov2_elvis_gld`, `dinov3_elvis_gld`, and `siglip2_elvis_gld`. 

You can download all models manually from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks) or use torch hub to use the specified model directly:

```
import torch

model = torch.hub.load('pavelsuma/elvis', 'dinov2_elvis_gld').eval()
```

Example usage of the model to estimate similarity between two input images is detailed in `demo.py`.


***

## Reproduction

### Evaluation
***
In order to evaluate the performance of our models, you need to have the extracted local descriptors of the datasets.
We provide them for ROxford5k, RParis6k. For other datasets, please see below how to extract them yourself.
The descriptors along with the extracted global similarities for the query nearest neighbors can be downloaded from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data).

You can also run the following command to download them into the `data` folder.
```
wget -r -nH --cut-dirs=5 --no-parent --reject="index.html*" -P data http://ptak.felk.cvut.cz/personal/sumapave/public/elvis/data/
```

A sample command to run the evaluation on these two datasets is as follows:

```
python3 -u elvis/evaluate.py --multirun \
        exp_name="dinov2_elvis_gld" \
        descriptors=dinov2 \
        data_root=data \
        model_path=dinov3_elvis_gld.pt \
        dataset@test_dataset="instre" \
        test_dataset.num_rerank=[400]
```


### Training
***

To train ELViS, you need to have the extracted local descriptors of the training set (GLDv2 or SOP).
DINOv2 local descriptors (float16) along with their computed global similarities can be downloaded from [here](http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data).
You can also run the following command to download them into the `data` folder.

```
wget -r -nH --cut-dirs=7 --no-parent --reject="index.html*" -P data/gldv2 http://ptak.felk.cvut.cz/personal/sumapave/public/ames/data/gldv2/
```

> Note: The training set is large and the download may take a while. You can extract the descriptors yourself by following the instructions below.

A sample command to train ELViS is as follows:

```
python3 -u elvis/train.py --multirun \
        descriptors=dinov2 \
        data_root=${PWD}/data \
        train_dataset.batch_size=600 \
```

***

## Extracting descriptors
***

Coming soon...


### Creating the nearest neighbor index

We provide the global-retrieval precomputed nearest neighbor indices for all datasets in files `nn_dinov2.pkl`, `nn_dinov3.pkl`, and `nn_siglip2.pkl` for the three respective backbones.
To reproduce this index creation using the extracted global descriptors, you can run the following command:
```
python extract/prepare_topk_global.py --dataset [dataset_name] --desc_name [dinov2|dinov3|siglip2] --data_root ${PWD}/data
```

### Combining multiple hdf5 files

As the number of local features is large for some datasets, it is beneficial to extract the features in parallel chunks and/or store the chunks in individual files. We provide a script that virtually links these chunks into a single hdf5 file for ease of use.
```
python extract/merge_hdf5.py --dataset [dataset_name] --desc_name [dinov2|dinov3|siglip2] --data_root ${PWD}/data
```


## Citation
***

```
@InProceedings{Suma_2026_ICLR,
    author    = {Suma, Pavel and Kordopatis-Zilos, Giorgos and Kalantidis, Yannis and Tolias, Giorgos},
    title     = {ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2026}
}
```

## Acknowledgements

This code is based on the repository of RRT:
[Instance-level Image Retrieval using Reranking Transformers](https://github.com/uvavision/RerankingTransformer).

CVNet extraction code is based on the repository of CVNet:
[Correlation Verification for Image Retrieval](https://github.com/sungonce/CVNet)