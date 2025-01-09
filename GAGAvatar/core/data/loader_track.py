#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import json
import torch
import pickle
import random
import numpy as np
import torchvision
from copy import deepcopy

from core.libs.utils_lmdb import LMDBEngine
from core.libs.flame_model import FLAMEModel

FOCAL_LENGTH = 12.0

class TrackedData(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split, cross_id=False):
        super().__init__()
        # build path
        self._split = split
        self._cross_id = cross_id
        assert self._split in ['train', 'val', 'test'], f'Invalid split: {self._split}'
        # meta data
        self._data_path = data_cfg.PATH
        self._point_plane_size = data_cfg.POINT_PLANE_SIZE
        # build records
        with open(os.path.join(self._data_path, 'optim.pkl'), 'rb') as f:
            self._data = pickle.load(f)
        with open(os.path.join(self._data_path, 'dataset.json'), 'r') as f:
            self._frames = json.load(f)[self._split]
        self._video_info, self._video_mapping = build_video_info(self._frames, cross_video=self._cross_id)
        if self._split in ['val', 'test']:
            first_frame = [self._video_info[v][0] for v in self._video_info.keys()]
            self._frames = [f for f in self._frames if f not in first_frame]
        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=data_cfg.FLAME_SCALE, no_lmks=True)

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(os.path.join(self._data_path, 'img_lmdb'), write=False)

    def _choose_image(self, frame_key, number=1):
        video_id = get_video_id(frame_key)
        if self._cross_id:
            video_id = self._video_mapping[video_id]
        if self._split == 'train':
            candidate_key = [key for key in self._video_info[video_id] if key != frame_key]
            feature_key = random.sample(candidate_key, k=number)[0]
        else:
            feature_key = self._video_info[video_id][0]
        f_image = self._lmdb_engine[feature_key].float() / 255.0
        f_shape = torch.tensor(self._data[feature_key]['shapecode']).float()
        f_transform = torch.tensor(self._data[feature_key]['transform_matrix']).float()
        f_planes = build_points_planes(self._point_plane_size, f_transform)
        return feature_key, f_image, f_shape, f_planes

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        # feature image
        f_key, f_image, f_shape, f_planes = self._choose_image(frame_key)
        f_image = torchvision.transforms.functional.resize(f_image, (518, 518), antialias=True)
        # driven image
        t_record = {}
        t_image = self._lmdb_engine[frame_key].float() / 255.0
        for key in ['bbox', 'posecode', 'shapecode', 'expcode', 'eyecode', 'transform_matrix']:
            t_record[key] = torch.tensor(self._data[frame_key][key]).float()
        t_points = self.flame_model(
            shape_params=t_record['shapecode'][None], pose_params=t_record['posecode'][None],
            expression_params=t_record['expcode'][None], eye_pose_params=t_record['eyecode'][None],
        )[0].float()
        one_record = {
            'f_image': f_image, 'f_shape': f_shape, 'f_planes': f_planes, 
            't_image': t_image, 't_points': t_points, 't_transform': t_record['transform_matrix'], 't_bbox': t_record['bbox'], 
            'infos': {'f_key':f_key, 't_key':frame_key},
        }
        return one_record

class TrackedData_audio(torch.utils.data.Dataset):
    def __init__(self, data_cfg, driver_data, t_images, t_transform, split, cross_id=False):
        super().__init__()
        # 保留原有变量
        self._split = split
        self._cross_id = cross_id
        assert self._split in ['train', 'val', 'test'], f'Invalid split: {self._split}'
        self._data_path = data_cfg.PATH
        self._point_plane_size = data_cfg.POINT_PLANE_SIZE

        # 根据DriverData_audio调整
        if isinstance(driver_data, dict) and 'audio' in driver_data:
            self._data = driver_data['audio']
            self._frames = list(range(len(self._data)))
        else:
            raise ValueError("driver_data must be a dictionary with an 'audio' key containing the data.")

        # meta data
        self.t_transform = t_transform
        self.t_images = t_images
        # 构建视频信息
        self._video_info, self._video_mapping = build_video_info(self._frames, cross_video=self._cross_id)
        
        # 按split划分frames
        if self._split in ['val', 'test']:
            first_frame = [self._video_info[v][0] for v in self._video_info.keys()]
            self._frames = [f for f in self._frames if f not in first_frame]

        # 构建模型
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=data_cfg.FLAME_SCALE, no_lmks=True)

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self):
        return len(self._frames)

    def _init_lmdb_database(self):
        self._lmdb_engine = LMDBEngine(os.path.join(self._data_path, 'img_lmdb'), write=False)

    def _choose_image(self, frame_key, number=1):
        video_id = get_video_id(frame_key)
        if self._cross_id:
            video_id = self._video_mapping[video_id]
        if self._split == 'train':
            candidate_key = [key for key in self._video_info[video_id] if key != frame_key]
            feature_key = random.sample(candidate_key, k=number)[0]
        else:
            feature_key = self._video_info[video_id][0]
        f_image = self._lmdb_engine[feature_key].float() / 255.0
        f_shape = torch.tensor(self._data[feature_key]['shapecode']).float()
        f_transform = torch.tensor(self._data[feature_key]['transform_matrix']).float()
        f_planes = build_points_planes(self._point_plane_size, f_transform)
        return feature_key, f_image, f_shape, f_planes

    def _load_one_record(self, frame_key):
        t_points = torch.tensor(self._data[frame_key]).float()
        t_image = self.t_images[frame_key]
        one_record = {
            't_image': t_image, 't_points': t_points, 't_transform': self.t_transform,
            'infos': {'t_key': frame_key},
        }
        return one_record

class DriverData(torch.utils.data.Dataset):
    def __init__(self, driver_path, feature_data=None, point_plane_size=296): # 确实没生成t_points，而是在_load_one_record中
        super().__init__()
        if type(driver_path) == str:
            self.driver_path = driver_path
            # build records
            self._is_video = True
            _records_path = os.path.join(self.driver_path, 'smoothed.pkl') # 可以没有pkl文件，但是必须有'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix'
            if not os.path.exists(_records_path):
                self._is_video = False
                _records_path = os.path.join(self.driver_path, 'optim.pkl')
            with open(_records_path, 'rb') as f:
                self._data = pickle.load(f) # 存了所有帧图的'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix'
                self._frames = sorted(list(self._data.keys()), key=lambda x:int(x.split('_')[-1])) # 帧图的序号
            if not self._is_video:
                self.shuffle_slice(60)
        else:
            self._is_video = False
            self._data = driver_path
            self._frames = list(self._data.keys())
            self._lmdb_engine = {key: self._data[key]['image']*255.0 for key in self._data.keys()}
        # meta data
        self.feature_data = feature_data # 源图像
        self.point_plane_size = point_plane_size
        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True)
        # build feature data
        if feature_data is None:
            _lmdb_engine = LMDBEngine(os.path.join(self.driver_path, 'img_lmdb'), write=False)
            frame_key = random.choice(self._frames)
            _f_image = _lmdb_engine[frame_key].float() / 255.0
            self.f_image = torchvision.transforms.functional.resize(_f_image, (518, 518), antialias=True)
            f_transform = torch.tensor(self._data[frame_key]['transform_matrix']).float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = torch.tensor(self._data[frame_key]['shapecode']).float().cpu()
            _lmdb_engine.close()
        else:
            self.f_image = torchvision.transforms.functional.resize(self.feature_data['image'].cpu(), (518, 518), antialias=True)
            f_transform = self.feature_data['transform_matrix'].float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = self.feature_data['shapecode'].float().cpu()

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def shuffle_slice(self, slice_num):
        import time
        import random
        random.seed(time.time())
        random.shuffle(self._frames)
        self._frames = self._frames[:slice_num]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        # print('Init the LMDB Database!')
        self._lmdb_engine = LMDBEngine(os.path.join(self.driver_path, 'img_lmdb'), write=False)

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        this_record = self._data[frame_key]
        for key in this_record.keys():
            if isinstance(this_record[key], np.ndarray):
                this_record[key] = torch.tensor(this_record[key])
        # ---audio-driven begin---
        t_image = self._lmdb_engine[frame_key].float() / 255.0 # 驱动图像
        t_points = self.flame_model( # （audio）只需要替换此处
            shape_params=self.f_shape[None], pose_params=this_record['posecode'][None],
            expression_params=this_record['expcode'][None], eye_pose_params=this_record['eyecode'][None],
        )[0].float()
        one_data = {
            'f_image': deepcopy(self.f_image), 'f_planes': deepcopy(self.f_planes), 
            't_image': t_image, 't_points': t_points, 't_transform': this_record['transform_matrix'], 
            'infos': {'t_key':frame_key},
        }
        # ---audio-driven end---
        return one_data

class DriverData_audio(torch.utils.data.Dataset):
    def __init__(self, driver_data, t_images, t_transform, feature_data=None, point_plane_size=296):
        super().__init__()
        if isinstance(driver_data, dict) and 'audio' in driver_data:
            self._is_video = True
            self._data = driver_data['audio']
            self._frames = list(self._data.keys())  # 使用第1维的长度作为帧的键
            # audio_data = driver_data['audio']
            # self._data = []
            # frame_num = len(self._frames)
            # for i in range(frame_num):
            #     frame_dict = {
            #         'exp': audio_data['exp'][:, i, :],
            #         'pose': audio_data['pose'][:, i, :],
            #         'shape': audio_data['shape'][:, i, :]
            #     }
            #     self._data.append(frame_dict)

        else:
            raise ValueError("driver_data must be a dictionary with an 'audio' key containing the data.")

        # meta data
        self.feature_data = feature_data
        self.point_plane_size = point_plane_size
        self.t_transform = t_transform  # 使用输入的t_transform
        self.t_images = t_images  # 新增的t_images参数
        # build model
        self.flame_model = FLAMEModel(n_shape=100, n_exp=50, scale=4.6, no_lmks=True)
        # build feature data
        if feature_data is None:
            raise ValueError("feature_data must be provided when using dictionary input.")
        else:
            self.f_image = torchvision.transforms.functional.resize(self.feature_data['image'].cpu(), (518, 518), antialias=True)
            f_transform = self.feature_data['transform_matrix'].float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = self.feature_data['shapecode'].float().cpu()

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def shuffle_slice(self, slice_num):
        import time
        import random
        random.seed(time.time())
        random.shuffle(self._frames)
        self._frames = self._frames[:slice_num]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        # This method is not used in the current context
        pass

    def _load_one_record(self, frame_key):
        this_record = {
            'exp': self._data[frame_key]['exp'],
            'pose': self._data[frame_key]['pose'],
            'shape': self._data[frame_key]['shape']
        }  
        for key in this_record.keys():
            if isinstance(this_record[key], np.ndarray):
                this_record[key] = torch.tensor(this_record[key])

        t_points = self.flame_model(
            shape_params=this_record['shape'][None], pose_params=this_record['pose'][None],
            expression_params=this_record['exp'][None]
        )[0].float()
        t_image = self.t_images[frame_key]  # 从t_images中获取对应的t_image
        one_data = {
            'f_image': deepcopy(self.f_image), 'f_planes': deepcopy(self.f_planes), 
            't_image': t_image,  # 使用对应的t_image
            't_points': t_points, 't_transform': self.t_transform,  # 使用输入的t_transform
            'infos': {'t_key': frame_key},
        }
        return one_data

class DriverData_audio_imitator(torch.utils.data.Dataset):
    def __init__(self, driver_data, t_images, t_transform, feature_data=None, point_plane_size=296):
        super().__init__()
        if isinstance(driver_data, dict) and 'audio' in driver_data:
            self._is_video = True
            self._data = driver_data['audio']
            self._frames = list(range(len(self._data)))  # 使用索引作为帧的键
        else:
            raise ValueError("driver_data must be a dictionary with an 'audio' key containing the data.")

        # meta data
        self.feature_data = feature_data
        self.point_plane_size = point_plane_size
        self.t_transform = t_transform  # 使用输入的t_transform
        self.t_images = t_images  # 新增的t_images参数
        # build model
        self.flame_model = FLAMEModel(n_shape=100, n_exp=50, scale=5.0, no_lmks=True)
        # build feature data
        if feature_data is None:
            raise ValueError("feature_data must be provided when using dictionary input.")
        else:
            self.f_image = torchvision.transforms.functional.resize(self.feature_data['image'].cpu(), (518, 518), antialias=True)
            f_transform = self.feature_data['transform_matrix'].float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = self.feature_data['shapecode'].float().cpu()

    def slice(self, slice):
        self._frames = self._frames[:slice]

    def shuffle_slice(self, slice_num):
        import time
        import random
        random.seed(time.time())
        random.shuffle(self._frames)
        self._frames = self._frames[:slice_num]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)

    def _init_lmdb_database(self):
        # This method is not used in the current context
        pass

    def _load_one_record(self, frame_key):
        t_points = torch.tensor(self._data[frame_key]).float()  # 从预测数据中获取t_points
        t_image = self.t_images[frame_key]  # 从t_images中获取对应的t_image
        one_data = {
            'f_image': deepcopy(self.f_image), 'f_planes': deepcopy(self.f_planes), 
            't_image': t_image,  # 使用对应的t_image
            't_points': t_points, 't_transform': self.t_transform,  # 使用输入的t_transform
            'infos': {'t_key': frame_key},
        }
        return one_data


def build_points_planes(plane_size, transforms):
    """
    构建点平面

    参数:
    plane_size (int): 平面的大小
    transforms (torch.Tensor): 变换矩阵

    返回:
    dict: 包含平面点和方向的字典
    """
    x, y = torch.meshgrid(
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        indexing="xy",
    )
    R = transforms[:3, :3]; T = transforms[:3, 3:]
    cam_dirs = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
    ray_dirs = torch.nn.functional.pad(
        torch.stack([x/FOCAL_LENGTH, y/FOCAL_LENGTH], dim=-1), (0, 1), value=1.0
    )
    cam_dirs = torch.matmul(R, cam_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    ray_dirs = torch.matmul(R, ray_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(ray_dirs.shape).squeeze()
    distance = ((origins[0] * cam_dirs[0]).sum()).abs()
    plane_points = origins + distance * ray_dirs
    return {'plane_points': plane_points, 'plane_dirs': cam_dirs[0]}


def build_video_info(frames, cross_video=False):
    video_info = {}
    for key in frames:
        video_id = get_video_id(key)
        if video_id not in video_info.keys():
            video_info[video_id] = []
        video_info[video_id].append(key)
    for video_id in video_info.keys():
        video_info[video_id] = sorted(
            video_info[video_id], key=lambda x:int(x.split('_')[-1])
        )
    video_mapping = {}
    if cross_video:
        video_ids = list(video_info.keys())
        video_ids = sorted(video_ids)
        for idx, video_id in enumerate(video_ids):
            if idx < len(video_ids) - 1:
                video_mapping[video_id] = video_ids[idx+1]
            else:
                video_mapping[video_id] = video_ids[0]
    return video_info, video_mapping


def get_video_id(frame_key):
    if frame_key.split('_')[0] in ['img']:
        video_id = frame_key.split('_')[1]
    else:
        video_id = frame_key.split('_')[0] 
    return video_id
