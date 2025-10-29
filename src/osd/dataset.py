import os
import torchaudio

data = torchaudio.datasets.LibriMix(
    root="/data/workspace/dataset/LibriMix",
    subset="test",
    num_speakers=3,
    sample_rate=16_000,
)

print("data:")
print(data[0])

print("meta_data")
print(data.get_metadata(0))

print("len(data):", len(data))
