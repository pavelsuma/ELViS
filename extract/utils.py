from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def batching(tensor, batch_sz):
    L = len(tensor)
    for i in range(L // batch_sz + 1):
        if i * batch_sz < L:
            yield tensor[i * batch_sz : (i + 1) * batch_sz]


def collate_batches(batch):
    aspect_ratios = defaultdict(list)
    # images, row_ids = zip(*batch)
    for image, row_id in batch:
        D, H, W = image[0].shape
        aspect_ratios[(H, W)].append((image, row_id))
    all_images, all_row_ids = [], []
    for (_), v in aspect_ratios.items():
        images, row_ids = zip(*v)
        images = [torch.stack(tensors, dim=0) for tensors in list(zip(*images))]
        all_images.append(images)
        all_row_ids.append(np.asarray(row_ids).squeeze())
    return all_images, all_row_ids


def load_image_pil(path):
    img = Image.open(path)
    img = img.convert("RGB")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        pass
    return img


def resize_image_pil(scale_x, scale_y, img):
    if scale_x != 1.0 or scale_y != 1.0:
        im = img.resize((round(scale_x * img.size[0]), round(scale_y * img.size[1])))
    else:
        im = img
    return im


def quantization_factor(side, scale, ps):
    if not ps:
        return scale

    new_side = scale * side
    quantize_to = max(round(new_side / ps), 1.0)
    return scale / ((new_side / ps) / quantize_to)
