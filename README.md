# 介绍

我们基于[GAGAvatar](https://github.com/xg-chu/GAGAvatar)开发了可语音驱动的3D数字人模型。
如下图所示![pipeline](https://github.com/shawnFuu/talkinggaga/blob/main/assets/pipeline.png).

我们的方法包含两个分支：重建模块与表情驱动模块。我们使用双提升方法（dual-lifting）生成头部高斯从而得到初步结果，然后使用神经渲染器（neural renderer）获得优化结果。我们利用[DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk)强大的生成能力将语音映射为FLAME模型参数，从而驱动头部口型变化。

# 环境

```
conda env create -f /data/fuxiaowen/talkinggaga/environment.yml
```
talkinggaga/GAGAvatar/diff-gaussian-rasterization需要按GAGAvatar指引安装

# Run

运行talkinggaga/GAGAvatar/test.py
