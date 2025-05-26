# 细胞分类模型

本项目基于 PyTorch 框架，构建了一个用于细胞分类的深度学习模型（阳性 / 阴性二分类），并提供了训练与测试脚本。

## 项目简介

该模型使用 MobileNetV3-Small 作为主干网络，结合数据增强与两阶段优化器策略（Adam + SGD），适用于医疗影像中小样本场景下的细胞分类任务。


## 项目结构

├── train.py # 用于模型训练
├── test.py # 使用训练好的模型进行预测测试
├── data/ # 图像数据目录（建议本地准备）
├── datafile/ # 包含数据划分的 txt 文件（train/val）
├── model_weights/ # 保存训练好的模型权重
├── tensorboard_logs/ # tensorboard 日志目录（可选）
├── README.md # 项目说明文件
└── .gitignore # 忽略无关文件



## 所需环境

请使用如下 Python 环境：

- Python 3.6
- PyTorch 1.9.0 + CUDA 11.1



## 使用说明

1. 训练模型
请确保准备好训练数据和标注 txt 文件（路径结构为 label;image_path）
python train.py
训练日志将保存在 tensorboard_logs/，模型权重保存在 model_weights/。

2. 测试模型
使用测试脚本对单张或多张图像进行推理：
python test.py --img path/to/image.jpg --weights model_weights/your_model.pth



## 注意事项

本项目未包含原始数据，请根据需要准备数据集，并与 train_cleanest_sample.txt / val_cleanest.txt 匹配路径格式。

推荐添加 .gitignore 忽略日志、模型权重和缓存文件。

建议使用 GPU 加速，默认会自动检测 cuda。
