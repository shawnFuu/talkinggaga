import subprocess
import os

def run_demo():
    script_path = "/data/fuxiaowen/talkinggaga/DiffPoseTalk/demo.py"
    exp_name = "head-SA-hubert-WM"
    iteration = "110000"
    audio_file = "demo/input/audio/FAST.flac"
    coef_file = "demo/input/coef/TH217.npy"
    style_file = "demo/input/style/TH217.npy"
    output_file = "TH217-FAST-TH217.mp4"
    n_repetitions = "3"
    cfg_scale_style = "3"
    cfg_scale_audio = "1.15"
    dynamic_threshold_ratio = "0.99"
    
    command = [
        "python", script_path,
        "--exp_name", exp_name,
        "--iter", iteration,
        "-a", audio_file,
        "-c", coef_file,
        "-s", style_file,
        "-o", output_file,
        "-n", n_repetitions,
        "-ss", cfg_scale_style,
        "-sa", cfg_scale_audio,
        "-dtr", dynamic_threshold_ratio
    ]
    subprocess.run(command)

if __name__ == "__main__":
    run_demo()
