from torch import nn

from .binarization_layer import BinarizationLayer

_BASE_URL = 'http://ptak.felk.cvut.cz/personal/sumapave/public/'


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)


class Matcher(nn.Module):
    _registry = {}

    def __init__(self, desc_name, local_dim, model_dim, binarized=True):
        super(Matcher, self).__init__()
        self.binarized = binarized

        if self.binarized:
            self.remap_local = nn.Sequential(
                BinarizationLayer(pretrained=_BASE_URL + f'ames/networks/itq_{desc_name}_D128.pt', trainable=True),
                nn.Linear(model_dim, model_dim),
                nn.LayerNorm(model_dim))
        else:
            self.remap_local = nn.Sequential(nn.Linear(local_dim, model_dim), nn.LayerNorm(model_dim))
        self.apply(init_weights)

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    def build_model(name, **kwargs):
        if name not in Matcher._registry:
            raise ValueError(f"Model '{name}' not found. Available: {list(Matcher._registry.keys())}")
        return Matcher._registry[name](**kwargs)