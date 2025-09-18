# 项目说明

本项目实现了一个基于 VAD 和非流式 ASR 的说话人识别系统，能够在多说话人环境下识别出不同说话人的身份。

## 使用方法

1. 克隆代码库到本地
2. 进入代码库目录
3. cd scripts 进入脚本目录
4. 运行 `bash install.sh` 安装依赖和下载模型, 模型在 models 目录下, 数据集在 dataset 目录下
5. 运行 `bash run.sh` 运行主程序
6. 运行 `bash test.sh` 运行测试脚本, 生成测试结果在 test 目录下

## 目录结构

### 脚本

在 scripts 目录下包含以下脚本：

1. install.sh 用于安装依赖, 模型, 数据集等
2. run.sh 用于运行主程序
3. generate-speaker-text.sh 用于生成说话人文本
4. split_speakers.py 用于分割 train 和 test 数据集
5. version.py 用于打印当前环境的版本信息
6. test.sh 用于 benchmark 测试

### 主程序

speaker-identification-with-vad-non-streaming-asr.py
