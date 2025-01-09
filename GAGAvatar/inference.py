#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
import argparse
import lightning
import numpy as np
import torchvision
from tqdm import tqdm
import sys

from predict_from_audio import predict_from_audio
from core.data.loader_track import DriverData, DriverData_audio
from core.models import build_model
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines.engine_core import CoreEngine as TrackEngine

def inference(image_path, driver_path, resume_path, force_retrack=False, device='cuda'):
    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith('/') else driver_path
    driver_name = os.path.basename(driver_path).split('.')[0]
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    model.eval()
    print(str(meta_cfg))
    track_engine = TrackEngine(focal_length=12.0, device=device)
    # build input data
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    # 包含'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix', 'image', 'vis_image'
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    # build driver data
    ### ------------ run on demo or tracked images/videos ---------- ###
    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
        driver_dataset = DriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE) # 进去看一下如何初始化
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    else:
        driver_name = os.path.basename(driver_path).split('.')[0]
        driver_data = get_tracked_results(driver_path, track_engine, force_retrack=force_retrack) # 看包含什么，如何得到
        if driver_data is None:
            print(f'Finish inference, no face in driver: {image_path}.')
            return
        driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE) # 'driver_path', '_is_video', '_data', '_frames', 'feature_data', 'point_plane_size', 'flame_model', 'f_image', 'f_planes', 'f_shape'
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    ### --------- if you need to run on your images online ---------- ###
    # driver_data = track_engine.track_image(your_images, your_image_names) # list of tensor, list of str
    # driver_dataset = DriverData(driver_data, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
    # driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)

    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    # run inference process
    # _water_mark_size = (82, 256)
    # _water_mark = torchvision.io.read_image('demos/gagavatar_logo.png', mode=torchvision.io.ImageReadMode.RGB_ALPHA).float()/255.0
    # _water_mark = torchvision.transforms.functional.resize(_water_mark, _water_mark_size, antialias=True).to(device)
    images = []

    ### ----------- if you need to run multiview results ------------ ###
    # view_angles = np.linspace(angle, -angle, frame_num)
    # batch['t_transform'] = build_camera(view_angles[idx])

    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_expression(batch) # 'f_image', 'f_planes', 't_image', 't_points', 't_transform', 'infos'
        gt_rgb = render_results['t_image'].clamp(0, 1)
        # pred_rgb = render_results['gen_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
        # pred_sr_rgb = add_water_mark(pred_sr_rgb, _water_mark)
        visulize_rgbs = torchvision.utils.make_grid([gt_rgb[0], pred_sr_rgb[0]], nrow=4, padding=0)
        images.append(visulize_rgbs.cpu())
    # dump_dir = os.path.join('render_results', meta_cfg.MODEL.NAME.split('_')[0])
    dump_dir = "/data/fuxiaowen/gaga/render_results/GAGAvatar"
    os.makedirs(dump_dir, exist_ok=True)
    if driver_dataset._is_video:
        dump_path = os.path.join(dump_dir, f'{driver_name}_{feature_name}.mp4')
        merged_images = torch.stack(images)
        feature_images = torch.stack([feature_data['image']]*merged_images.shape[0])
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(dump_path, merged_images, fps=25.0)
    else:
        dump_path = os.path.join(dump_dir, f'{driver_name}_{feature_name}.jpg')
        # 将图像拼接
        merged_images = torchvision.utils.make_grid(images, nrow=5, padding=0)
        feature_images = torchvision.utils.make_grid([feature_data['image']]*(merged_images.shape[-2]//512), nrow=1, padding=0)
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        torchvision.utils.save_image(merged_images, dump_path)
    print(f'Finish inference: {dump_path}.')

def inference_audio(image_path, audio_path, coef_path, style_path, output_path, resume_path, force_retrack=False, device='cuda'):
    lightning.fabric.seed_everything(42)
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    model.eval()
    print(str(meta_cfg))
    track_engine = TrackEngine(focal_length=12.0, device=device)
    # build input data
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    t_transform = feature_data['transform_matrix']
    # 包含'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix', 'image', 'vis_image'
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    # 获取音频驱动的t_points预测
    coef_dict, t_images = predict_from_audio(audio_path,coef_path,style_path,output_path) # exp [rep_n,frame_num,50] pose [rep_n,frame_num,6] shape [rep_n,frame_num,100]

    # 构建driver数据
    for rep_idx, current_coef_dict in enumerate(coef_dict):
        # 构建driver数据
        print(f"generate video {rep_idx}......")
        if len(coef_dict) == 1:
            t_images_rep = t_images.squeeze(0)
        else:
            t_images_rep = t_images[rep_idx].squeeze(0)
        driver_dataset = DriverData_audio({'audio': current_coef_dict}, t_images_rep, t_transform, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=1, shuffle=False)

        driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
        # run inference process
        images = []
        for idx, batch in enumerate(tqdm(driver_dataloader)):
            render_results = model.forward_expression_audio(batch) # 'f_image', 'f_planes', 't_points', 't_transform', 'infos'

            pred_sr_rgb = render_results['sr_gen_image']
            gt_rgb = render_results['t_image']

            # 保存gt和预测图像
            gt_rgb_resized = torch.nn.functional.interpolate(gt_rgb.float().unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
            visulize_rgbs = torchvision.utils.make_grid([gt_rgb_resized, pred_sr_rgb[0]], nrow=4, padding=0)
            # 保存到save_dir
            images.append(visulize_rgbs.cpu())
        dump_dir = "/data/fuxiaowen/gaga/render_results/GAGAvatar"
        os.makedirs(dump_dir, exist_ok=True)
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        dump_path = os.path.join(dump_dir, f'{feature_name}_{audio_name}_{rep_idx}.mp4')
        merged_images = torch.stack(images)
        feature_images = torch.stack([feature_data['image']]*merged_images.shape[0])
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        # 先生成无音频的视频
        video_no_audio = os.path.join(dump_dir, f'{feature_name}_{audio_name}_no_audio_{rep_idx}.mp4')
        torchvision.io.write_video(video_no_audio, merged_images, fps=25.0)
        # 添加音频并生成最终视频
        add_audio_to_video(audio_path, video_no_audio, dump_path)
    print(f'Finish inference: {dump_path}.')


def inference_audio_imitator(image_path, audio_path, resume_path, force_retrack=False, device='cuda'):
    lightning.fabric.seed_everything(42)
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    model.eval()
    print(str(meta_cfg))
    track_engine = TrackEngine(focal_length=12.0, device=device)
    # build input data
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    t_transform = feature_data['transform_matrix']
    # 包含'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix', 'image', 'vis_image'
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    # 获取音频驱动的t_points预测
    t_points_predictions, t_images = predict_from_audio(audio_path)

    # 缩放尺度
    scale = 4.7
    t_points_predictions = t_points_predictions * scale

    # 构建driver数据
    driver_dataset = DriverData_audio({'audio': t_points_predictions}, t_images,t_transform, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)

    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    # run inference process
    images = []
    for idx, batch in enumerate(tqdm(driver_dataloader)):
    # for idx, batch in enumerate(driver_dataloader):
        render_results = model.forward_expression_audio(batch) # 'f_image', 'f_planes', 't_points', 't_transform', 'infos'

        # gt_rgb = render_results['t_image'].clamp(0, 1)
        # pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image']
        gt_rgb = render_results['t_image']

        # 保存gt和预测图像
        visulize_rgbs = torchvision.utils.make_grid([gt_rgb, pred_sr_rgb[0]], nrow=4, padding=0)
        # 保存到save_dir
        images.append(visulize_rgbs.cpu())
    dump_dir = "/data/fuxiaowen/gaga/render_results/GAGAvatar"
    os.makedirs(dump_dir, exist_ok=True)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    dump_path = os.path.join(dump_dir, f'{feature_name}_{audio_name}.mp4')
    merged_images = torch.stack(images)
    feature_images = torch.stack([feature_data['image']]*merged_images.shape[0])
    merged_images = torch.cat([feature_images, merged_images], dim=-1)
    merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
    # 先生成无音频的视频
    video_no_audio = os.path.join(dump_dir, f'{feature_name}_{audio_name}_no_audio.mp4')
    torchvision.io.write_video(video_no_audio, merged_images, fps=25.0)
    # 添加音频并生成最终视频
    add_audio_to_video(audio_path, video_no_audio, dump_path)
    print(f'Finish inference: {dump_path}.')


def add_audio_to_video( audio_file, video_file, out_vid_w_audio_file):

    print("Audio file", audio_file)
    ffmpeg_command = f"ffmpeg -y -i {video_file} -i {audio_file} -map 0:v -map 1:a -c:v copy -shortest {out_vid_w_audio_file}"
    os.system(ffmpeg_command)
    print("added audio to the video", out_vid_w_audio_file)

    if sys.platform.startswith('win'):
        rm_command = ('del "{0}" '.format(video_file))
        print(rm_command)
    else:
        rm_command = ('rm {0} '.format(video_file))
    
    os.system(rm_command)
    print("removed ", video_file)


def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
    tracked_pt_path = 'render_results/tracked/tracked.pt'
    if not os.path.exists(tracked_pt_path):
        os.makedirs('render_results/tracked', exist_ok=True)
        torch.save({}, tracked_pt_path)
    tracked_data = torch.load(tracked_pt_path, weights_only=False)
    image_base = os.path.basename(image_path)
    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {tracked_pt_path}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path]) # 'bbox', 'shapecode', 'expcode', 'posecode', 'eyecode', 'transform_matrix', 'image', 'vis_image'
        if feature_data is not None:
            feature_data = feature_data[image_path]
            torchvision.utils.save_image(
                torch.tensor(feature_data['vis_image']), 'render_results/tracked/{}.jpg'.format(image_base.split('.')[0])
            )
        else:
            print(f'No face detected in {image_path}.')
            return None
        tracked_data[image_base] = feature_data
        # track all images in this folder
        # 似乎有点重复？
        other_names = [i for i in os.listdir(os.path.dirname(image_path)) if is_image(i)]
        other_paths = [os.path.join(os.path.dirname(image_path), i) for i in other_names]
        if len(other_paths) <= 35:
            print('Track on all images in this folder to save time.')
            other_images = [torchvision.io.read_image(imp, mode=torchvision.io.ImageReadMode.RGB).float() for imp in other_paths]
            try:
                other_feature_data = track_engine.track_image(other_images, other_names)
                for key in other_feature_data:
                    torchvision.utils.save_image(
                        torch.tensor(other_feature_data[key]['vis_image']), 'render_results/tracked/{}.jpg'.format(key.split('.')[0])
                    )
                tracked_data.update(other_feature_data)
            except Exception as e:
                print(f'Error: {e}.')
        # save tracking result
        torch.save(tracked_data, tracked_pt_path)
    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


def is_image(image_path):
    extension_name = image_path.split('.')[-1].lower()
    return extension_name in ['jpg', 'png', 'jpeg']


def add_water_mark(image, water_mark):
    _water_mark_rgb = water_mark[None, :3]
    _water_mark_alpha = water_mark[None, 3:4].expand(-1, 3, -1, -1) * 0.8
    _mark_patch = image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:]
    _mark_patch = _mark_patch * (1-_water_mark_alpha) + _water_mark_rgb * _water_mark_alpha
    image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:] = _mark_patch
    return image


### ------- multi-view camera helper -------- ###
def build_camera(angle, ori_transforms=None, device='cuda'):
    from pytorch3d.renderer.cameras import look_at_view_transform
    if ori_transforms is None:
        distance = 9.3
    else:
        distance = ori_transforms[..., 3].square().sum(dim=-1).sqrt()[0].item() * 1.0
        device = ori_transforms.device
    print(f'Camera distance: {distance}, angle: {angle}.')
    R, T = look_at_view_transform(distance, 5, angle, device=device) # D, E, A
    rotate_trans = torch.cat([R, T[:, :, None]], dim=-1)
    return rotate_trans


### ------------ run speed test ------------- ###
def speed_test():
    driver_path = './demos/vfhq_driver'
    resume_path = './assets/GAGAvatar.pt'
    lightning.fabric.seed_everything(42)
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator='cuda', strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    print(str(meta_cfg))
    # build driver data
    driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
    driver_dataset = DriverData(driver_path, None, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    # run inference process
    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_expression(batch)
        gt_rgb = render_results['t_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
    print(f'Finish speed test.')
    # torchvision.utils.save_image([gt_rgb[0], pred_sr_rgb[0]], 'speed_test.jpg')


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--force_retrack', '-f', action='store_true')
    parser.add_argument('--resume_path', '-r', default='./assets/GAGAvatar.pt', type=str)
    parser.add_argument('--gaga', action='store_true', default=False)
    
    args, unknown = parser.parse_known_args()
    
    if args.gaga:
        parser.add_argument('--audio_path', '-a', required=True, type=str)
        parser.add_argument('--coef_path', '-c', required=True, type=str)
        parser.add_argument('--style_path', '-s', required=True, type=str)
        parser.add_argument('--output_path', '-o', required=True, type=str)
    else:
        parser.add_argument('--driver_path', '-d', required=True, type=str)
    
    args = parser.parse_args()
    
    # launch
    torch.set_float32_matmul_precision('high')
    if args.gaga:
        inference_audio(args.image_path, args.audio_path, args.coef_path, args.style_path, args.output_path, args.resume_path, args.force_retrack)
    else:
        inference(args.image_path, args.driver_path, args.resume_path, args.force_retrack)