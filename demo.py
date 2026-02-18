import torch
import hydra
from torchvision.transforms import functional as TF, transforms
from omegaconf import DictConfig

from extract.spatial_attention_2d import SpatialAttention2d
from extract.utils import load_image_pil, quantization_factor
from extract.extract_descriptors import load_model
from extract.extract import get_local
from elvis.models import matcher, ames, elvis

_BASE_URL = 'http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks/'

def load_image(img_path, imsize, patch_size, transform):
    img = TF.to_tensor(load_image_pil(img_path))
    scale = imsize / max(img.shape)

    scale_x = quantization_factor(img.shape[2], scale, patch_size)
    scale_y = quantization_factor(img.shape[1], scale, patch_size)
    im = TF.resize(img, size=[round(scale_y * img.shape[-2]), round(scale_x * img.shape[-1])],
                   interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    im = transform(im)
    return im

def get_local_features(image, backbone, detector, num_local):
    feats, _ = backbone(image[None].cuda())[-1]
    _, weights = detector(feats)
    local_feats, _, _, _, _ = get_local([feats], [weights], num_local)
    return local_feats

@torch.inference_mode()
@hydra.main(config_path="conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.cpu else 'cpu')
    imsize = 770
    num_local = 700

    backbone, mean, std = load_model(cfg.descriptors.model_name)
    backbone.to(device).eval()
    transform = transforms.Normalize(mean=mean, std=std)

    detector = SpatialAttention2d(cfg.descriptors.dim_local_features)
    detector.cuda()
    detector.eval()
    cpt = torch.hub.load_state_dict_from_url(f'{_BASE_URL}/{cfg.descriptors.desc_name}_detector.pt')
    detector.load_state_dict(cpt['state'], strict=True)

    model = matcher.Matcher.build_model(desc_name=cfg.descriptors.desc_name, local_dim=cfg.descriptors.dim_local_features,
                                        **cfg.model, pretrained=f'{cfg.descriptors.desc_name}_{cfg.model.name}.pt')
    model.to(device).eval()

    image_paths = ['001.jpg', '005.jpg']
    imgs = [load_image(im, imsize, cfg.descriptors.patch_size, transform) for im in image_paths]

    img_1_feat = get_local_features(imgs[0], backbone, detector, num_local)
    img_2_feat = get_local_features(imgs[1], backbone, detector, num_local)

    sim = model(src_local=img_1_feat, tgt_local=img_2_feat)
    
    print(f"Similarity score: {sim.item()}")

if __name__ == "__main__":
    main()
