from elvis.models.ames import AMES
from elvis.models.elvis import Elvis

dependencies = ["torch"]

def dinov2_elvis(*, pretrained: bool = True, **kwargs):
    return Elvis(desc_name='dinov2', local_dim=768, pretrained='dinov2_elvis.pt' if pretrained else None, binarized=False, **kwargs)

def dinov2_ames(*, pretrained: bool = True, **kwargs):
    return AMES(desc_name='dinov2', local_dim=768, pretrained='dinov2_ames.pt' if pretrained else None, binarized=False, **kwargs)

def dinov2_ames_dist(*, pretrained: bool = True, **kwargs):
    return AMES(desc_name='dinov2', local_dim=768, pretrained='dinov2_ames_dist.pt' if pretrained else None, binarized=True, **kwargs)

def cvnet_ames(*, pretrained: bool = True, **kwargs):
    return AMES(desc_name='cvnet', local_dim=1024, pretrained='cvnet_ames.pt' if pretrained else None, binarized=False, **kwargs)

def cvnet_ames_dist(*, pretrained: bool = True, **kwargs):
    return AMES(desc_name='cvnet', local_dim=1024, pretrained='cvnet_ames_dist.pt' if pretrained else None, binarized=True, **kwargs)
