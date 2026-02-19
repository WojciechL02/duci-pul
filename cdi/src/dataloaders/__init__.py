from src.dataloaders.dit_dataloader import get_dit_dataloader

from torch.utils.data import DataLoader
from typing import Dict

try:
    from src.dataloaders.uvit_t2i_dataloader import get_uvit_t2i_dataloaders
except ImportError:
    get_uvit_t2i_dataloaders = None


loaders: Dict[str, DataLoader] = {
    "dit": get_dit_dataloader,
}
if get_uvit_t2i_dataloaders is not None:
    loaders["coco_uvit"] = get_uvit_t2i_dataloaders