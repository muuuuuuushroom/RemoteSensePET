from .backbones import *
from .transformer import *
from .position_encoding import *
from .pet import build_pet

def build_model(args):
    return build_pet(args)

