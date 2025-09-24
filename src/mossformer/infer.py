import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from src.mossformer.dataset import Libri2Mix8kDataset

ds = Libri2Mix8kDataset.load_test()
print(ds[1])  # type: ignore
input = ds[1]["mix_wav:FILE"]  # type: ignore
# input可以是url也可以是本地文件路径
# input = "https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav"
separation = pipeline(
    Tasks.speech_separation,
    model="iic/speech_mossformer_separation_temporal_8k",
    device="cuda:0",
)

result = separation(input)

for i, signal in enumerate(result["output_pcm_list"]):  # type: ignore
    save_file = f"../../test-mossformer/output_spk{i}.wav"
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
