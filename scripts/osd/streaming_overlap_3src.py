#!/usr/bin/env python3
"""
Streaming version of offline_overlap_3src.py for real-time microphone input
"""
import argparse
import time
import pyaudio
import numpy as np
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import List
from streaming_overlap3_core import StreamingOverlap3Pipeline


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 流式参数
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size")
    p.add_argument(
        "--process-seconds",
        type=float,
        default=2.0,
        help="Seconds of audio to process each time",
    )

    # 目标说话人（必需）
    p.add_argument(
        "--target-wav", required=True, help="Enrollment audio for target speaker"
    )

    # OSD参数
    p.add_argument("--osd-backend", default="pyannote")
    p.add_argument("--osd-thr", type=float, default=0.5)
    p.add_argument("--osd-win", type=float, default=0.5)
    p.add_argument("--osd-hop", type=float, default=0.1)

    # 分离参数
    p.add_argument("--sep-backend", default="asteroid")
    p.add_argument("--sep-checkpoint", default="")

    # ASR参数
    p.add_argument("--paraformer", default="")
    p.add_argument("--sense-voice", default="")
    p.add_argument("--encoder", default="")
    p.add_argument("--decoder", default="")
    p.add_argument("--joiner", default="")
    p.add_argument("--tokens", default="")
    p.add_argument("--decoding-method", default="greedy_search")
    p.add_argument("--feature-dim", type=int, default=80)
    p.add_argument("--language", default="auto")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--provider", default="cpu")

    # 说话人验证参数
    p.add_argument(
        "--spk-embed-model", required=True, help="Speaker embedding ONNX model"
    )
    p.add_argument("--sv-threshold", type=float, default=0.6)

    # 重叠处理参数
    p.add_argument("--min-overlap-dur", type=float, default=0.4)

    # 输出参数
    p.add_argument("--output-dir", default="streaming_results")
    p.add_argument(
        "--save-interval",
        type=float,
        default=10.0,
        help="Save results interval in seconds",
    )

    return p.parse_args()


class StreamingApplication:
    """流式应用程序"""

    def __init__(self, args):
        self.args = args
        self.pipeline = StreamingOverlap3Pipeline(args, args.target_wav)

        # 音频采集参数
        self.chunk_size = args.chunk_size
        self.sample_rate = args.sample_rate
        self.frames_per_process = int(
            args.sample_rate * args.process_seconds / args.chunk_size
        )

        # 输出设置
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.running = False
        self.all_results = []
        self.threads: List[threading.Thread] = []

    def setup_audio(self):
        """设置音频采集"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    def start(self):
        """启动流式处理"""
        self.running = True
        self.setup_audio()

        """可能有底层c库冲突
        # 启动音频采集线程
        audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        audio_thread.start()
        
        # 启动结果处理线程
        result_thread = threading.Thread(target=self._result_processing_loop, daemon=True)
        result_thread.start()
        
        # 启动保存线程
        save_thread = threading.Thread(target=self._save_loop, daemon=True)
        save_thread.start()
        """
        # 启动音频采集线程
        audio_thread = threading.Thread(
            target=self._audio_capture_loop, name="audio_capture"
        )
        audio_thread.start()
        self.threads.append(audio_thread)
        # 启动结果处理线程
        result_thread = threading.Thread(
            target=self._result_processing_loop, name="result_processing"
        )
        result_thread.start()
        self.threads.append(result_thread)
        # 启动保存线程
        # save_thread = threading.Thread(target=self._save_loop, name="save_loop")
        # save_thread.start()
        # self.threads.append(save_thread)

        print("Streaming started... Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
        finally:
            # 确保退出时资源被清理
            if self.running:
                self.stop()

    def _audio_capture_loop(self):
        """音频采集循环"""
        audio_buffer = []

        while self.running:
            try:
                # 读取音频数据
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                audio_buffer.append(audio_array)

                # 达到处理长度时进行处理
                if len(audio_buffer) >= self.frames_per_process:
                    # 合并音频数据
                    combined_audio = np.concatenate(audio_buffer)
                    self.pipeline.add_audio_data(combined_audio)
                    audio_buffer = []

            except Exception as e:
                # print(f"Audio capture error: {e}")
                # time.sleep(0.1)
                print(f"Audio capture error / stream closed: {e}")
                break

    def _result_processing_loop(self):
        """结果处理循环"""
        while self.running:
            try:
                results = self.pipeline.get_results()
                for result in results:
                    print(
                        f"[{result['kind']}] Stream:{result.get('stream', '')} Text: {result['text']} "
                        f"(Score: {result.get('sv_score', 0):.3f})"
                    )
                    self.all_results.append(result)
                time.sleep(0.1)
            except Exception as e:
                print(f"Result processing error: {e}")
                time.sleep(0.1)

    def _save_loop(self):
        """定期保存结果"""
        last_save = time.time()

        while self.running:
            current_time = time.time()
            if current_time - last_save >= self.args.save_interval:
                self._save_results()
                last_save = current_time
            time.sleep(1)

    def _save_results(self):
        """保存结果到文件"""
        if not self.all_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"results_{timestamp}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for result in self.all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Results saved to {output_file}")

    def stop(self):
        """停止流式处理"""
        self.running = False

        # 先等待 audio_capture 线程退出（优先,不然会和底层c库报错）
        for t in self.threads:
            if t.name == "audio_capture":
                try:
                    print(f"等待 audio_capture 线程退出...")
                    t.join(timeout=2.0)
                    if t.is_alive():
                        print(
                            "audio_capture 未退出，尝试呼叫 stream.stop_stream() 以强制中断..."
                        )
                        try:
                            if hasattr(self, "stream") and self.stream is not None:
                                self.stream.stop_stream()
                        except Exception as e:
                            print(f"调用 stop_stream 时出错: {e}")
                        # 再次等待
                        t.join(timeout=1.0)
                except Exception as e:
                    print(f"等待 audio_capture 时出错: {e}")

        # 现在安全关闭音频资源
        try:
            if hasattr(self, "stream") and self.stream is not None:
                try:
                    self.stream.close()
                    print("音频流已关闭")
                except Exception as e:
                    print(f"关闭音频流时出错: {e}")
            if hasattr(self, "audio") and self.audio is not None:
                try:
                    self.audio.terminate()
                    print("PyAudio 已终止")
                except Exception as e:
                    print(f"终止 PyAudio 时出错: {e}")
        except Exception as e:
            print(f"关闭音频资源时出错: {e}")

        # 其它线程再 join
        print("等待其他线程结束...")
        for t in self.threads:
            if t.name != "audio_capture":
                try:
                    if t.is_alive():
                        print(f"等待线程 {t.name} 结束...")
                        t.join(timeout=1.0)
                        if t.is_alive():
                            print(f"警告: 线程 {t.name} 未正常结束")
                except Exception as e:
                    print(f"等待线程 {t.name} 时出错: {e}")

        # 处理缓冲区中剩余的数据
        print("Flushing remaining audio data...")
        self.pipeline.flush_buffer()

        # 最终保存
        print("Saving final results...")
        self._save_results()
        print("Streaming stopped.")


def main():
    args = parse_args()
    app = StreamingApplication(args)
    app.start()


if __name__ == "__main__":
    main()
