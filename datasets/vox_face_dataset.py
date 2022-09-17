'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-09-17 16:26:42
Email: haimingzhang@link.cuhk.edu.cn
Description: The VoxCeleb1 datset to load 3DMM parameters and the
corresponding rendering face.
'''


import os
import os.path as osp
import random
import collections
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image
from omegaconf import OmegaConf
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from .preprocess import FaceAligner


class VoxFaceDataset(Dataset):
    def __init__(self, opt, is_inference, cache=True):
        self.data_root = osp.join(opt.path, "train") \
            if not is_inference else osp.join(opt.path, "test")

        # self.video_items, self.person_ids, self.idx_by_person_id, self.video_frame_length_dict = \
        #     self._build_dataset(self.data_root)
        
        cache_path = osp.join(osp.dirname(opt.path), "vox1_cache.pkl")
        if cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data_info = pickle.load(f)
        else:
            data_info = self._build_dataset(self.data_root)
            if cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(data_info, f)
        
        self.video_items, self.person_ids, self.idx_by_person_id, self.video_frame_length_dict = data_info

        self.face_aligner = FaceAligner()

        self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _build_dataset(self, data_dir):
        all_videos_name = sorted(os.listdir(data_dir))
        person_ids = sorted(list({video.split('#')[0] for video in all_videos_name}))

        idx_by_persion_id_dict = collections.defaultdict(list)
        for index, video_item in enumerate(all_videos_name):
            person_id = video_item.split('#')[0]
            idx_by_persion_id_dict[person_id].append(index)
        
        ## Get the number of frames in each video
        video_frame_length_dict = {}
        for video_name in tqdm(all_videos_name):
            meta_info_path = osp.join(data_dir, video_name, 'meta_info.yaml')
            meta_info = OmegaConf.load(meta_info_path)
            video_frame_length_dict[video_name] = int(meta_info.num_frames)

        return all_videos_name, person_ids, idx_by_persion_id_dict, video_frame_length_dict

    def __getitem__(self, index):
        person_id = self.person_ids[index]
        video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
        frame_source, frame_target = self.random_select_frames(video_item)

        data = self._load_data(video_item, frame_source, frame_target)

        return data

    def _load_data(self, choose_video, frame_source, frame_target):
        data = {}

        video_images_dir = osp.join(self.data_root, choose_video, "face_image")

        reference_image_path = osp.join(video_images_dir, f"{frame_source:06d}.png")
        raw_reference_image = Image.open(reference_image_path).convert("RGB")

        ## Load the landmarks to align the face
        raw_source_lm = np.loadtxt(
            osp.join(video_images_dir, f"{frame_source:06d}.txt")).astype(np.float32)
        raw_source_lm[:, -1] = 256 - 1 - raw_source_lm[:, -1]

        _, source_image, _, _ = self.face_aligner.align_face(raw_reference_image, raw_source_lm)
        data['source_image'] = self.transform(source_image)

        ## =============== Load the target image ===============
        target_image_path = osp.join(video_images_dir, f"{frame_target:06d}.png")
        raw_target_image = Image.open(target_image_path).convert("RGB")
        
        raw_target_lm = np.loadtxt(
            osp.join(video_images_dir, f"{frame_target:06d}.txt")).astype(np.float32)
        raw_target_lm[:, -1] = 256 - 1 - raw_target_lm[:, -1]

        _, target_image, _, _ = self.face_aligner.align_face(raw_target_image, raw_target_lm)
        data['target_image'] = self.transform(target_image)

        ## Load the 3DMM parameters of target image
        face_3dmm_fp = osp.join(self.data_root, choose_video, "deep3dface", f"{frame_target:06d}.mat")
        face_3dmm = convert_3dmm(face_3dmm_fp)
        data['target_semantics'] = face_3dmm
        return data
    
    def __len__(self):
        return len(self.person_ids)

    def random_select_frames(self, video_item):
        num_frame = self.video_frame_length_dict[video_item]
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]


def convert_3dmm(file_path):
    file_mat = loadmat(file_path)
    coeff_3dmm = file_mat['coeff']
    crop_param = file_mat['transform_params']
    _, _, ratio, t0, t1 = np.hsplit(crop_param.astype(np.float32), 5)
    crop_param = np.concatenate([ratio, t0, t1], 1)
    coeff_3dmm_cat = np.concatenate([coeff_3dmm, crop_param], 1) 
    return coeff_3dmm_cat