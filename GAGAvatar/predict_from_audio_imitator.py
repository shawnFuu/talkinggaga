import os
import sys

from Imitator_raw.imitator.test.test_model_external_audio import test_on_audio
from Imitator_raw.imitator.test.test_model_voca import get_latest_checkpoint

# from Imitator.test_model_external_audio import test_on_audio
# from Imitator.test.test_model_voca import get_latest_checkpoint


def get_parser(parser):
    parser.add_argument('-g', "--gpus", type=str, default=0)
    parser.add_argument('-m', "--model", type=str, default="pretrained_model/generalized_model_mbp_vel")
    parser.add_argument('-o', "--out_dir", type=str, default=None)
    parser.add_argument('-a', "--audio", type=str, required=True)
    parser.add_argument('-t', '--test_subject', type=str, default="FaceTalk_170809_00138_TA")
    parser.add_argument('-c', '--test_condition', type=str, default="2")
    parser.add_argument('-d', '--dump_results', action='store_true')
    parser.add_argument('-r', '--render_results', action='store_true')
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.set_defaults(unseen=False)
    return parser


def predict_from_audio(audio_path):
    # 配置参数
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = get_parser(parser)
    opt = parser.parse_args([
        '-g', '0',
        '-m', '/home/fuxiaowen/3DGS/Imitator/pretrained_model/generalized_model_mbp_vel', 
        '-a', audio_path,
        '-t', 'FaceTalk_170915_00223_TA',
        '-c', '2',
        '-s', '23'
    ])

    # 初始化测试类
    tester = test_on_audio()
    best_ckpt = get_latest_checkpoint(os.path.join(opt.model, "checkpoints"))
    
    # 运行测试
    prediction, t_images = tester.run_test_on_the_audio_path(opt, opt.model, best_ckpt, gaga=True)
    
    return prediction, t_images
