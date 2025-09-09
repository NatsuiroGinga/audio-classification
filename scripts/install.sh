#!/bin/bash
# Install necessary dependencies
pip install sherpa-onnx==1.11.1+cuda -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
pip install soundfile
pip install sounddevice
pip install numpy
# conda 环境 importError XX/lib/libstdc++.so.6: version `GLIBCXX_3.4.30‘ not found 解决方法
# 需要改成自己的libstdc++.so.6路径
# ln -sf /data/workspace/llm/anaconda3/envs/audio/lib/libstdc++.so.6.0.30 /data/workspace/llm/anaconda3/envs/audio/lib/libstdc++.so.6

# Install CUDA 11.8
mkdir ../cuda
cd ../cuda || exit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

chmod +x cuda_11.8.0_520.61.05_linux.run

./cuda_11.8.0_520.61.05_linux.run \
  --silent \
  --toolkit \
  --installpath=/star-fj/fangjun/software/cuda-11.8.0 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page

# Install cuDNN for CUDA 11.8
wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz

tar xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz --strip-components=1 -C /star-fj/fangjun/software/cuda-11.8.0

# Set environment variables for CUDA 11.8
echo 'export CUDA_HOME=/star-fj/fangjun/software/cuda-11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export CUDA_TOOLKIT_ROOT=$CUDA_HOME
export CUDA_BIN_PATH=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
export CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS' > ./activate-cuda-11.8.sh

# Download models
mkdir -p ../models/vad
mkdir -p ../models/speaker-recongition
mkdir -p ../models/asr

echo 'Downloading models...'

echo 'Downloading silero_vad.onnx...'
wget -P ../models/vad https://github.com/snakers4/silero-vad/raw/refs/tags/v5.0/files/silero_vad.onnx

echo 'Downloading 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx...'
wget -P ../models/speaker-recongition speaker-recongition/3dspeaker_speech_eres2net_base https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo 'Downloading sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17...'
wget -P ../models/asr wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

echo 'All done!'
echo 'Show files in models directory:'
ls -R ../models

# Download dataset
mkdir -p ../dataset
echo 'Downloading dataset...'
wget -P ../dataset https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/test.tar.gz
tar xvf ../dataset/test.tar.gz -C ../dataset
rm ../dataset/test.tar.gz

# Split speakers into train and test sets
echo 'Splitting speakers into train and test sets...'

bash ./generate-speaker-text.sh

python ./split_speakers.py \
  --input ../dataset/speaker.txt \
  --train-out ../dataset/train-speaker.txt \
  --test-out ../dataset/test-speaker.txt \
  --train-ratio 0.8 \
  --seed 42 \
  --mode utterance

echo 'All done!'
echo 'Show files in dataset directory:'
ls -R ../dataset
