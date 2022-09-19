'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-18 19:16:56
Email: haimingzhang@link.cuhk.edu.cn
Description: The main trainer
'''

import argparse
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model_real_face import RealFaceModel
import yaml
import os
from datasets import get_dataloader


def parse_config():
    parser = argparse.ArgumentParser(description='The main trainer')
    parser.add_argument('-c', '--config', type=str, default='conf/vox_conf.yaml')

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    return cfg


if __name__ == '__main__':
    config = parse_config()

    model = RealFaceModel(cfg=config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'./check_points/',
        every_n_train_steps = 10000,
        save_on_train_epoch_end = True,
        save_last=True)

    if config.checkpoint_path is None:
        print(f"[WARNING] Train from scratch!")
    else:
        print(f"[WARNING] Load pretrained model from {config.checkpoint}")
        model = model.load_from_checkpoint(config.checkpoint_path, config=config)

    train_dataloader = get_dataloader(config.dataset)

    trainer = Trainer(
            accelerator='gpu', 
            devices=1,
            precision=16,
            callbacks=[checkpoint_callback],
            max_epochs=50)

    trainer.fit(model, train_dataloader, ckpt_path=config.checkpoint_path)
