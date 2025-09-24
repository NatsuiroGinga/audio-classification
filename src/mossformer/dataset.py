"""
数据格式
每条数据包含5个属性:

id 不重复的id字符串
mix_wav:FILE 混合音频文件路径
s1_wav:FILE 第一个讲话人的音频文件路径
s2_wav:FILE 第二个讲话人的音频文件路径
length 以帧数表示的文件长度
"""

from modelscope.msdatasets import MsDataset


class Libri2Mix8kDataset:
    dataset_name = "Libri2Mix_8k"

    @staticmethod
    def load_test():
        return MsDataset.load(
            dataset_name=Libri2Mix8kDataset.dataset_name, split="test"
        )

    @staticmethod
    def load_train():
        return MsDataset.load(
            dataset_name=Libri2Mix8kDataset.dataset_name, split="train"
        )

    @staticmethod
    def load_dev():
        return MsDataset.load(dataset_name=Libri2Mix8kDataset.dataset_name, split="dev")
