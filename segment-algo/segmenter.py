import numpy as np
import librosa
from scipy import signal  # (目前未使用，可考虑后续加入频谱特征)
from typing import List, Tuple
from funasr import AutoModel
from typing import List, Tuple
import soundfile as sf
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AudioSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 1024,
        hop_length: int = 512,
        energy_threshold: float = 0.01,
        silence_duration: float = 0.3,
        min_segment_length: float = 1.0,
        max_segment_length: float = 10.0,
    ):
        """
        快速音频断句器

        Args:
            sample_rate: 采样率
            frame_length: 帧长度
            hop_length: 跳跃长度
            energy_threshold: 能量阈值（相对于最大能量）
            silence_duration: 静音持续时间阈值（秒）
            min_segment_length: 最小段长度（秒）
            max_segment_length: 最大段长度（秒）
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length

    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """计算音频能量"""
        # 使用短时能量
        energy = librosa.feature.rms(
            y=audio, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        return energy

    def detect_silence_regions(self, energy: np.ndarray) -> List[Tuple[int, int]]:
        """检测静音区域"""
        # 归一化能量
        max_energy = np.max(energy)
        if max_energy == 0:
            return [(0, len(energy))]

        normalized_energy = energy / max_energy

        # 检测低能量区域
        silence_mask = normalized_energy < self.energy_threshold

        # 找到连续的静音区域
        silence_regions = []
        in_silence = False
        start_idx = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                start_idx = i
                in_silence = True
            elif not is_silent and in_silence:
                # 检查静音持续时间
                duration = (i - start_idx) * self.hop_length / self.sample_rate
                if duration >= self.silence_duration:
                    silence_regions.append((start_idx, i))
                in_silence = False

        # 处理结尾的静音
        if in_silence:
            duration = (
                (len(silence_mask) - start_idx) * self.hop_length / self.sample_rate
            )
            if duration >= self.silence_duration:
                silence_regions.append((start_idx, len(silence_mask)))

        return silence_regions

    def segment_audio(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        对音频进行断句分段

        Args:
            audio: 输入音频数组

        Returns:
            List of (start_sample, end_sample) tuples
        """
        # 计算能量
        energy = self.compute_energy(audio)

        # 检测静音区域
        silence_regions = self.detect_silence_regions(energy)

        # 基于静音区域生成分段
        segments = []
        last_end = 0

        for silence_start, silence_end in silence_regions:
            # 转换帧索引到样本索引
            segment_start_sample = last_end
            segment_end_sample = silence_start * self.hop_length

            # 检查段长度
            segment_duration = (
                segment_end_sample - segment_start_sample
            ) / self.sample_rate

            if segment_duration >= self.min_segment_length:
                # 如果段太长，进一步分割
                if segment_duration > self.max_segment_length:
                    sub_segments = self._split_long_segment(
                        segment_start_sample, segment_end_sample
                    )
                    segments.extend(sub_segments)
                else:
                    segments.append((segment_start_sample, segment_end_sample))

            last_end = silence_end * self.hop_length

        # 处理最后一段
        if last_end < len(audio):
            segment_duration = (len(audio) - last_end) / self.sample_rate
            if segment_duration >= self.min_segment_length:
                if segment_duration > self.max_segment_length:
                    sub_segments = self._split_long_segment(last_end, len(audio))
                    segments.extend(sub_segments)
                else:
                    segments.append((last_end, len(audio)))

        return segments

    def _split_long_segment(self, start: int, end: int) -> List[Tuple[int, int]]:
        """分割过长的段"""
        segments = []
        max_samples = int(self.max_segment_length * self.sample_rate)

        current_start = start
        while current_start < end:
            current_end = min(current_start + max_samples, end)
            segments.append((current_start, current_end))
            current_start = current_end

        return segments


# 使用示例
def process_audio_for_speaker_verification(audio_path: str, segmenter: AudioSegmenter):
    """处理音频进行声纹识别"""
    # 加载音频
    audio, sr = librosa.load(audio_path, sr=segmenter.sample_rate)

    # 分段
    segments = segmenter.segment_audio(audio)

    # 提取音频段
    audio_segments = []
    for start, end in segments:
        segment = audio[start:end]
        audio_segments.append(segment)

    return audio_segments, segments


# 初始化分段器
# segmenter = AudioSegmenter(
#     sample_rate=16000,
#     energy_threshold=0.02,  # 可调整
#     silence_duration=0.2,  # 200ms静音
#     min_segment_length=1.0,  # 最小1秒
#     max_segment_length=8.0  # 最大8秒
# )


class FunASRSegmenter:
    def __init__(
        self,
        device: str = "cpu",
        vad_model: str = "fsmn-vad",
        max_segment_length: float = 10.0,
        min_segment_length: float = 1.0,
        merge_threshold: float = 0.5,
    ):
        """
        基于FunASR VAD的音频分段器

        Args:
            vad_model: VAD模型名称
            max_segment_length: 最大段长度（秒）
            min_segment_length: 最小段长度（秒）
            merge_threshold: 合并阈值（秒）
        """
        self.vad_model = AutoModel(model=vad_model, device=device)
        self.max_segment_length = max_segment_length
        self.min_segment_length = min_segment_length
        self.merge_threshold = merge_threshold

    def segment_audio_with_vad(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        使用VAD进行音频分段

        Args:
            audio_path: 音频文件路径

        Returns:
            List of (start_time, end_time) in seconds
        """
        # 使用VAD检测语音活动
        vad_result = self.vad_model.generate(input=audio_path)

        if not vad_result or not vad_result[0]["value"]:
            return []

        # 获取VAD检测的时间段
        vad_segments = vad_result[0]["value"]  # [[start_ms, end_ms], ...]

        # 转换为秒并优化分段
        segments = []
        for start_ms, end_ms in vad_segments:
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            segments.append((start_sec, end_sec))

        # 优化分段：合并短间隔，分割长段
        optimized_segments = self._optimize_segments(segments)

        return optimized_segments

    def _optimize_segments(
        self, segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """优化分段结果"""
        if not segments:
            return []

        optimized = []
        current_start, current_end = segments[0]

        for i in range(1, len(segments)):
            next_start, next_end = segments[i]

            # 检查是否需要合并
            gap = next_start - current_end
            current_duration = current_end - current_start

            if (
                gap <= self.merge_threshold
                and current_duration + gap + (next_end - next_start)
                <= self.max_segment_length
            ):
                # 合并段
                current_end = next_end
            else:
                # 添加当前段（可能需要分割）
                if current_duration >= self.min_segment_length:
                    if current_duration > self.max_segment_length:
                        # 分割长段
                        sub_segments = self._split_long_segment(
                            current_start, current_end
                        )
                        optimized.extend(sub_segments)
                    else:
                        optimized.append((current_start, current_end))

                current_start, current_end = next_start, next_end

        # 处理最后一段
        current_duration = current_end - current_start
        if current_duration >= self.min_segment_length:
            if current_duration > self.max_segment_length:
                sub_segments = self._split_long_segment(current_start, current_end)
                optimized.extend(sub_segments)
            else:
                optimized.append((current_start, current_end))

        return optimized

    def _split_long_segment(
        self, start: float, end: float
    ) -> List[Tuple[float, float]]:
        """分割过长的段"""
        segments = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + self.max_segment_length, end)
            segments.append((current_start, current_end))
            current_start = current_end

        return segments

    def extract_audio_segments(self, audio_path: str):
        """提取音频段"""
        segments = self.segment_audio_with_vad(audio_path)

        # 加载完整音频
        audio, sr = sf.read(audio_path)

        audio_segments = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment = audio[start_sample:end_sample]
            audio_segments.append(segment)

        return audio_segments, segments


# 使用示例
# funasr_segmenter = FunASRSegmenter(
#     max_segment_length=8.0,
#     min_segment_length=1.5,
#     merge_threshold=0.3
# )


# ---------------------------------------------------------------------------
# 轻量级流式句级分段（基于能量/自适应噪声门限 + 可选 webrtcvad）
# 适用场景：希望在生成或采集音频时快速得到较短的“句子”/发话单元，避免单纯停顿阈值造成超长段。
# 核心特性：
#   1. 自适应噪声底噪估计（前若干帧分位数）
#   2. 双阈值（进入/退出）+ 最小语音 & 最小静音时长
#   3. 目标句长 (target_sentence_sec) 超过后，回溯最近窗口寻找“软”能量谷值进行切分
#   4. 硬上限 (hard_max_sentence_sec) 强制切分，保证不会形成过长片段
#   5. 可选 webrtcvad 叠加（若安装），用 AND 方式提高句边界精度（可配置）
# 输出：与其它分段器保持接口一致 -> extract_audio_segments(audio_path) -> (List[np.ndarray], List[Tuple[start_sec, end_sec]])
# ---------------------------------------------------------------------------

try:  # 可选依赖
    import webrtcvad  # type: ignore

    _HAS_WEBRTCVAD = True
except Exception:  # noqa
    _HAS_WEBRTCVAD = False


@dataclass
class StreamingSegmenterConfig:
    sample_rate: int = 16000
    frame_ms: int = 20  # 10/20/30 ms (webrtcvad 只支持这三种)
    hop_ms: int = 20  # 可设置 < frame_ms 实现重叠；此处保持等于帧长简化
    min_speech_sec: float = 0.25
    min_silence_sec: float = 0.20
    start_threshold_db: float = -40.0  # 进入语音与噪声底相比的 dB 相对阈值
    end_threshold_db: float = -45.0  # 退出语音阈值（更低）
    target_sentence_sec: float = 3.5  # 期望平均句长（超过尝试软切分）
    hard_max_sentence_sec: float = 5.0  # 绝对最大长度（强制切分）
    soft_search_window_sec: float = 1.0  # 软切分回溯窗口
    vad_aggressiveness: int = 2  # 0-3（仅在 webrtcvad 可用时）
    use_webrtcvad: bool = True  # 若安装则启用
    combine_with_vad: bool = True  # True: 能量 & VAD 双判决；False: 只用能量
    energy_smooth: int = 3  # 简单滑动平均平滑窗口（帧数）
    noise_init_sec: float = 0.5  # 前多少秒估计噪声底
    energy_eps: float = 1e-8


class StreamingSentenceSegmenter:
    """轻量级流式句级分段器（离线也可一次性处理）。

    算法要点：
        - 逐帧计算 RMS -> dBFS
        - 前 noise_init_sec 建立噪声底分布 (20/50/80 百分位)，动态阈值= noise_p50 + start_threshold_db 等
        - 状态机（SILENCE -> SPEECH）
        - 语音内超过 target_sentence_sec，向后延续直到找到软谷值或到达硬上限
    """

    def __init__(self, config: StreamingSegmenterConfig | None = None):
        self.config = config or StreamingSegmenterConfig()
        self._validate()
        self._maybe_init_vad()

    def _validate(self):
        if self.config.frame_ms not in (10, 20, 30):
            logger.warning("frame_ms 不是 10/20/30，将影响 webrtcvad，可选改为 20")
        if self.config.hard_max_sentence_sec <= self.config.target_sentence_sec:
            raise ValueError("hard_max_sentence_sec 必须大于 target_sentence_sec")

    def _maybe_init_vad(self):
        if _HAS_WEBRTCVAD and self.config.use_webrtcvad:
            try:
                self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
            except Exception as e:  # noqa
                logger.warning(f"初始化 webrtcvad 失败，禁用: {e}")
                self.vad = None
        else:
            self.vad = None

    def _frame_bytes(self, frame: np.ndarray) -> bytes:
        # 期望 int16 mono
        if frame.dtype != np.int16:
            # 裁剪再量化
            clipped = np.clip(frame, -1.0, 1.0)
            frame_i16 = (clipped * 32767).astype(np.int16)
        else:
            frame_i16 = frame
        return frame_i16.tobytes()

    def _compute_energy_db(
        self, audio: np.ndarray, sr: int, frame_len: int, hop: int
    ) -> np.ndarray:
        # 简单逐帧 RMS
        n_frames = 1 + max(0, (len(audio) - frame_len) // hop)
        energies = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + frame_len]
            if len(frame) < frame_len:
                pad = np.zeros(frame_len - len(frame), dtype=frame.dtype)
                frame = np.concatenate([frame, pad])
            rms = np.sqrt(np.mean(frame**2) + self.config.energy_eps)
            db = 20 * np.log10(rms + self.config.energy_eps)
            energies.append(db)
        energies = np.array(energies, dtype=np.float32)
        if self.config.energy_smooth > 1:
            k = self.config.energy_smooth
            kernel = np.ones(k) / k
            energies = np.convolve(energies, kernel, mode="same")
        return energies

    def segment(self, audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        cfg = self.config
        frame_len = int(sr * cfg.frame_ms / 1000)
        hop = int(sr * cfg.hop_ms / 1000)
        if hop <= 0:
            hop = frame_len
        energies_db = self._compute_energy_db(audio, sr, frame_len, hop)
        n_frames = len(energies_db)

        # 噪声统计
        init_frames = max(1, int((cfg.noise_init_sec * sr - frame_len) // hop) + 1)
        noise_sample = energies_db[:init_frames]
        noise_p20 = np.percentile(noise_sample, 20)
        noise_p50 = np.percentile(noise_sample, 50)
        noise_floor = noise_p20
        start_th = noise_p50 + (cfg.start_threshold_db)
        end_th = noise_p50 + (cfg.end_threshold_db)

        # 边界集合
        segments: List[Tuple[int, int]] = []
        state = "SILENCE"
        current_start_frame = None
        silence_count = 0
        min_speech_frames = int(cfg.min_speech_sec * 1000 / cfg.hop_ms)
        min_silence_frames = int(cfg.min_silence_sec * 1000 / cfg.hop_ms)
        target_frames = int(cfg.target_sentence_sec * 1000 / cfg.hop_ms)
        hard_max_frames = int(cfg.hard_max_sentence_sec * 1000 / cfg.hop_ms)
        soft_search_frames = int(cfg.soft_search_window_sec * 1000 / cfg.hop_ms)

        # 预取 webrtcvad 帧布尔（若启用）
        vad_flags = None
        if self.vad is not None and cfg.combine_with_vad:
            vad_flags = []
            # webrtcvad 需要 16bit, 10/20/30ms；若 hop != frame_len 这里仍按帧长做
            for i in range(n_frames):
                start = i * hop
                frame = audio[start : start + frame_len]
                if len(frame) < frame_len:
                    pad = np.zeros(frame_len - len(frame), dtype=frame.dtype)
                    frame = np.concatenate([frame, pad])
                ok = False
                try:
                    ok = self.vad.is_speech(self._frame_bytes(frame), sr)
                except Exception:  # noqa
                    ok = energies_db[i] > start_th  # 回退
                vad_flags.append(ok)
            vad_flags = np.array(vad_flags, dtype=bool)

        def is_speech_frame(idx: int) -> bool:
            energy_flag = (
                energies_db[idx] > start_th
                if state == "SILENCE"
                else energies_db[idx] > end_th
            )
            if vad_flags is None:
                return energy_flag
            if cfg.combine_with_vad:
                # 进入语音需双真；内部保持时放宽为 OR (避免被 VAD 短暂打断)
                if state == "SILENCE":
                    return energy_flag and vad_flags[idx]
                else:
                    return energy_flag or vad_flags[idx]
            else:
                return energy_flag or vad_flags[idx]

        for f in range(n_frames):
            speech_flag = is_speech_frame(f)
            if state == "SILENCE":
                if speech_flag:
                    state = "SPEECH"
                    current_start_frame = f
                    silence_count = 0
            else:  # SPEECH
                if speech_flag:
                    silence_count = 0
                else:
                    silence_count += 1
                current_len = f - (current_start_frame or f)

                # 尝试软截断
                if current_len >= target_frames:
                    # 回溯窗口找最低能量阈值点
                    search_start = max(
                        (current_start_frame or 0) + min_speech_frames,
                        f - soft_search_frames,
                    )
                    if search_start < f - min_silence_frames:  # 确保有足够窗口
                        window_idx = np.arange(search_start, f)
                        window_energy = energies_db[window_idx]
                        # 目标：能量低于 (noise_floor + 5dB) 的局部极小值
                        candidate_mask = window_energy < (noise_floor + 5)
                        if np.any(candidate_mask):
                            sub_idx = window_idx[candidate_mask]
                            # 取能量最小点
                            split_frame = int(sub_idx[np.argmin(energies_db[sub_idx])])
                            # 需保证最小语音长度
                            if (
                                split_frame - (current_start_frame or 0)
                                >= min_speech_frames
                            ):
                                seg_start_sample = (current_start_frame or 0) * hop
                                seg_end_sample = split_frame * hop
                                segments.append((seg_start_sample, seg_end_sample))
                                current_start_frame = split_frame
                                # 重置长度窗口继续
                                continue

                # 硬截断
                if current_len >= hard_max_frames:
                    seg_start_sample = (current_start_frame or 0) * hop
                    seg_end_sample = f * hop
                    segments.append((seg_start_sample, seg_end_sample))
                    state = "SILENCE"
                    current_start_frame = None
                    silence_count = 0
                    continue

                # 正常结束（静音满足）
                if (
                    silence_count >= min_silence_frames
                    and current_len >= min_speech_frames
                ):
                    seg_start_sample = (current_start_frame or 0) * hop
                    # 回退静音部分作为结束
                    seg_end_frame = f - silence_count
                    seg_end_sample = seg_end_frame * hop
                    segments.append((seg_start_sample, seg_end_sample))
                    state = "SILENCE"
                    current_start_frame = None
                    silence_count = 0

        # 尾部收尾
        if state == "SPEECH" and current_start_frame is not None:
            end_sample = len(audio)
            start_sample = current_start_frame * hop
            if (end_sample - start_sample) / sr >= self.config.min_speech_sec:
                segments.append((start_sample, end_sample))

        # 过滤空/过短
        cleaned = []
        min_len_samples = int(self.config.min_speech_sec * sr)
        for s, e in segments:
            if e - s >= min_len_samples:
                cleaned.append((s, e))
        return cleaned

    # 与其它分段器接口统一
    def extract_audio_segments(self, audio_path: str):
        audio, sr = sf.read(audio_path)
        if sr != self.config.sample_rate:
            # 重采样保证统一（避免后续保存强制 16k 出现时长偏差）
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.config.sample_rate
            )
            sr = self.config.sample_rate
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        segments_samples = self.segment(audio, sr)
        seg_arrays = [audio[s:e] for s, e in segments_samples]
        # 转换为秒区间
        segments_time = [(s / sr, e / sr) for s, e in segments_samples]
        return seg_arrays, segments_time
