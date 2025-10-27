import os
import torchaudio

data = torchaudio.datasets.LibriMix(
    root=".",
    subset="dev",
    num_speakers=3,
    sample_rate=16000,
)

print(data[0])  # 打印第一个样本的信息
