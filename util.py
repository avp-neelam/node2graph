import random
import torch

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed:int=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_name(name:str) -> str:
    name = name.lower()
    aliases = {
        'cham': 'chameleon',
        'chameleon': 'chameleon',
        'cora': 'cora',
        'citeseer': 'citeseer',
        'pubmed': 'pubmed',
        'texas': 'texas',
        'cornell': 'cornell',
        'wisconsin': 'wisconsin',
    }
    if name not in aliases:
        raise ValueError(f"Unknown dataset alias: {name}")
    return aliases[name]