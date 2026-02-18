from .chamfer import MatchCS
from .ames import AMES
from .elvis import ELViS
from .matcher_rrt import MatchERT
from .matcher_sf import MatchSF


def get_model(cfg):
    if cfg.name == 'ames':
        return AMES
    elif cfg.name == 'elvis':
        return ELViS