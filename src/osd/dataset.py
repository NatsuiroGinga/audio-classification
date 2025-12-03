import os
import torchaudio

# 英文, 3说话人混合数据集
# data = torchaudio.datasets.LibriMix(
#     root="/data/workspace/dataset/LibriMix",
#     subset="test",
#     num_speakers=3,
#     sample_rate=16_000,
# )

# print("data:")
# print(data[0])

# print("meta_data")
# print(data.get_metadata(0))

# print("len(data):", len(data))

# 中文
from modelscope.msdatasets import MsDataset

ds = MsDataset.load("speech_tts/AISHELL-3", subset_name="default", split="test")

print("len(ds):", len(ds))
print(ds[0])
