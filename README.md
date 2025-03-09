# 介绍

我们基于GAGAvatar开发了可语音驱动的数字人模型，主要思路为借助DiffPoseTalk输入音频生成Flame参数，其余流程遵循GAGAvatar的pipeline。

# 环境

先按照environment.yml安装conda环境
```
conda env create -f /data/fuxiaowen/talkinggaga/environment.yml
```
talkinggaga/GAGAvatar/diff-gaussian-rasterization需要按GAGAvatar指引安装

psbody-mesh==0.4需要clone原仓库安装，参考https://blog.csdn.net/weixin_51352126/article/details/134614580

然后下载并把资源放置在相应的位置

talkinggaga_asset/DiffPoseTalk\data放置在talkinggaga/DiffPoseTalk/models

talkinggaga_asset/DiffPoseTalk/DPT和talkinggaga_asset/DiffPoseTalk/SE放置在talkinggaga/DiffPoseTalk/experiments

talkinggaga_asset/DiffPoseTalk/HDTF_TFHP放置在talkinggaga/DiffPoseTalk/datasets

talkinggaga_asset/GAGAvatar/assets放置在talkinggaga/GAGAvatar

然后请检查代码文件所有带fuxiaowen的地方，替换成你自己的文件夹路径

# Run

运行talkinggaga/GAGAvatar/test.py即可
