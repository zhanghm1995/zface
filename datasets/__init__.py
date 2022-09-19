import torch

from .vox_face_dataset import VoxFaceDataset


def get_dataset(cfg):
    if cfg.type == 'VoxFaceDataset':
        return VoxFaceDataset(**cfg.params)
    else:
        raise NotImplementedError


def get_dataloader(cfg):
    dataset = get_dataset(cfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **cfg.dataloader_params)
    return dataloader
