#!/usr/bin/env python3
"""合并原始测试数据和扩充的谐音词数据。

将 kws_test_data 和 kws_test_data_expanded 合并为完整的测试集。
"""

import json
import shutil
from pathlib import Path
from collections import Counter


def main():
    root = Path(__file__).parent.parent.parent
    original_dir = root / "dataset/kws_test_data"
    expanded_dir = root / "dataset/kws_test_data_expanded"
    merged_dir = root / "dataset/kws_test_data_merged"

    # 加载原始数据
    print("加载原始测试数据...")
    with open(original_dir / "metadata.json", "r") as f:
        original_meta = json.load(f)

    # 加载扩充数据
    print("加载扩充测试数据...")
    with open(expanded_dir / "metadata.json", "r") as f:
        expanded_meta = json.load(f)

    # 创建合并目录
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "positive").mkdir(exist_ok=True)
    (merged_dir / "negative").mkdir(exist_ok=True)

    # 合并正样本（只复制原始的）
    print("\n复制正样本...")
    positive_samples = []
    for sample in original_meta["positive_samples"]:
        src = Path(sample["file"])
        dst = merged_dir / "positive" / src.name
        shutil.copy2(src, dst)
        sample_copy = sample.copy()
        sample_copy["file"] = str(dst)
        positive_samples.append(sample_copy)

    # 合并负样本
    print("合并负样本...")
    negative_samples = []

    # 复制原始负样本
    for sample in original_meta["negative_samples"]:
        src = Path(sample["file"])
        dst = merged_dir / "negative" / src.name
        shutil.copy2(src, dst)
        sample_copy = sample.copy()
        sample_copy["file"] = str(dst)
        negative_samples.append(sample_copy)

    # 复制扩充的谐音词样本
    for sample in expanded_meta["negative_samples"]:
        src = Path(sample["file"])
        # 重命名避免冲突
        dst = merged_dir / "negative" / f"expanded_{src.name}"
        shutil.copy2(src, dst)
        sample_copy = sample.copy()
        sample_copy["file"] = str(dst)
        negative_samples.append(sample_copy)

    # 生成合并后的 metadata
    merged_meta = {
        "timestamp": expanded_meta["timestamp"],
        "config": original_meta["config"],
        "statistics": {
            "total_positive_files": len(positive_samples),
            "total_negative_files": len(negative_samples),
            "original_negative": len(original_meta["negative_samples"]),
            "expanded_negative": len(expanded_meta["negative_samples"]),
        },
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
    }

    # 保存 metadata
    with open(merged_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    # 统计谐音词分布
    print("\n✓ 合并完成!")
    print(f"  正样本: {len(positive_samples)}")
    print(f"  负样本: {len(negative_samples)}")
    print(f"    - 原始: {len(original_meta['negative_samples'])}")
    print(f"    - 扩充: {len(expanded_meta['negative_samples'])}")

    phrases = [s.get("phrase", "unknown") for s in negative_samples]
    phrase_counts = Counter(phrases)
    print(f"\n谐音词数量（前10）:")
    for phrase, count in sorted(phrase_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {phrase}: {count}")

    print(f"\n合并数据集保存到: {merged_dir}")


if __name__ == "__main__":
    main()
