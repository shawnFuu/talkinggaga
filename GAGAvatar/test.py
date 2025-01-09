import subprocess # 调试应用subprocess模块运行命令
import os

def main():
    # 检查当前工作路径

    # 提前提供参数
    image_path = "/data/fuxiaowen/talkinggaga/GAGAvatar/demos/examples/19.jpg"
    driver_path = "/data/fuxiaowen/talkinggaga/GAGAvatar/demos/drivers/obama"
    resume_path = "/data/fuxiaowen/talkinggaga/GAGAvatar/assets/GAGAvatar.pt"
    force_retrack = False
    audio_path = "/data/fuxiaowen/talkinggaga/DiffPoseTalk/demo/input/audio/cxk.mp3"
    coef_path = "/data/fuxiaowen/talkinggaga/DiffPoseTalk/demo/input/coef/Obama.npy"
    style_path = "/data/fuxiaowen/talkinggaga/DiffPoseTalk/demo/input/style/head-L4H4-T0.1-BS32/iter_0026000/TH217.npy"
    output_path = "/data/fuxiaowen/talkinggaga/GAGAvatar/outputs/TH217-FAST-TH217.mp4"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    command = [
        "python", "/data/fuxiaowen/talkinggaga/GAGAvatar/inference.py",
        "--image_path", image_path,
        "--audio_path", audio_path,
        "--coef_path", coef_path,
        "--style_path", style_path,
        "--output_path", output_path,
        "--resume_path", resume_path,
        "--gaga"
    ]
    
    # command = [
    #     "python", "/home/fuxiaowen/3DGS/GAGAvatar/inference.py",
    #     "--image_path", image_path,
    #     # "--audio_path", audio_path,
    #     "--driver_path", driver_path,
    #     "--resume_path", resume_path,
    #     # "--gaga"
    # ]
    
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
