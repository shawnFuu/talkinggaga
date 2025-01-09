import importlib.util
import argparse

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

demo = import_module_from_path('demo', '/data/fuxiaowen/talkinggaga/DiffPoseTalk/demo.py')
Demo = demo.Demo
infer_from_file_audio = Demo.infer_from_file_audio

from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description='DiffTalkingHead: Speech-Driven 3D Facial Animation using diffusion model')
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--iter', type=int, default=1000000, help='number of iterations')
    parser.add_argument('--coef_stats', type=str, default='/home/fuxiaowen/3DGS/DiffPoseTalk/datasets/HDTF_TFHP/lmdb/stats_train.npz',
                        help='path to the coefficient statistics')

    # Inference
    parser.add_argument('--mode', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--black_bg', action='store_true', help='whether to use black background')
    parser.add_argument('--no_context_audio_feat', action='store_true',
                        help='whether to use only the current audio feature')
    parser.add_argument('--dynamic_threshold_ratio', '-dtr', type=float, default=0,
                        help='dynamic thresholding ratio. 0 to disable')
    parser.add_argument('--dynamic_threshold_min', '-dtmin', type=float, default=1.)
    parser.add_argument('--dynamic_threshold_max', '-dtmax', type=float, default=4.)
    parser.add_argument('--save_coef', action='store_true', help='whether to save the generated coefficients')

    parser.add_argument('--audio', '-a', type=Path, required=True, help='path of the input audio signal')
    parser.add_argument('--coef', '-c', type=Path, required=True, help='path to the coefficients')
    parser.add_argument('--style', '-s', type=Path, help='path to the style feature')
    parser.add_argument('--tex', '-t', type=Path, help='path of the rendered video')
    parser.add_argument('--no_head', action='store_true', help='whether to include head pose')
    parser.add_argument('--output', '-o', type=Path, required=True, help='path of the rendered video')
    parser.add_argument('--n_repetitions', '-n', type=int, default=1, help='number of repetitions')
    parser.add_argument('--scale_audio', '-sa', type=float, default=1.15, help='guiding scale')
    parser.add_argument('--scale_style', '-ss', type=float, default=3, help='guiding scale')
    parser.add_argument('--cfg_mode', type=str, choices=['incremental', 'independent'])
    parser.add_argument('--cfg_cond', type=str)
    return parser

def predict_from_audio(audio_path, coef_path, style_path, output_path):
    # 配置参数
    parser = get_parser()
    args = parser.parse_args([
        '--exp_name', 'SA-hubert-WM',
        '--iter', '100000',
        '--audio', audio_path,
        '--coef', coef_path,
        '--style', style_path,
        '--output', output_path,
        '--n_repetitions', '1',
        '--scale_style', '3',
        '--scale_audio', '1.15',
        '--dynamic_threshold_ratio', '0.99'
    ])

    # 创建Demo实例
    demo_app = Demo(args)

    # 设置配置条件和比例
    cfg_cond = demo_app.model.guiding_conditions if args.cfg_cond is None else args.cfg_cond.split(',')
    cfg_scale = []
    for cond in cfg_cond:
        if cond == 'audio':
            cfg_scale.append(args.scale_audio)
        elif cond == 'style':
            cfg_scale.append(args.scale_style)

    # 运行推理
    coef_dict, rendered_images = demo_app.infer_from_file_audio(
        audio_path=args.audio,
        coef_path=args.coef,
        out_path=args.output,
        style_path=args.style,
        tex_path=args.tex,
        n_repetitions=args.n_repetitions,
        ignore_global_rot=args.no_head,
        cfg_mode=args.cfg_mode,
        cfg_cond=cfg_cond,
        cfg_scale=cfg_scale
    )
    
    # 将coef_dict转换为frame_num个子字典
    rep_num = coef_dict['exp'].shape[0]
    frame_num = coef_dict['exp'].shape[1]
    transformed_coef_list = []

    for rep_idx in range(rep_num):
        rep_dict = {}
        for frame_idx in range(frame_num):
            rep_dict[frame_idx] = {
                'exp': coef_dict['exp'][rep_idx, frame_idx, :].cpu(),
                'pose': coef_dict['pose'][rep_idx, frame_idx, :].cpu(),
                'shape': coef_dict['shape'][rep_idx, frame_idx, :].cpu()
            }
        transformed_coef_list.append(rep_dict)

    coef_dict = transformed_coef_list
    return coef_dict, rendered_images
