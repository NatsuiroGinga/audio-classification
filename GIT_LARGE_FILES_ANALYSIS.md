# Git 提交历史中的大文件分析报告

## 📊 发现摘要

**是的，git 提交历史中存在大文件。**

检测到 **1 个大文件** 被提交到 git 仓库：

| 文件名 | 大小 | 提交时间 | 文件类型 | 状态 |
|--------|------|---------|---------|------|
| `core.1021870` | **3.6 GB** | 2026-01-05 18:56:22 | 核心转储/二进制文件 | ❌ 被追踪（应忽略）|

---

## 🔍 详细分析

### 1. 大文件信息

```
文件名: core.1021870
大小: 3.6 GB (3,6G)
权限: -rw-r--r--
最后修改: Jan 9 10:40
```

### 2. Git 提交历史

- **提交哈希**: `07811849a608cd3a7e22b1d57e28b750e0ad3e14`
- **提交日期**: Mon Jan 5 18:56:22 2026 +0800
- **提交者**: NatsuiroGinga <duyueyeweiliang@gmail.com>
- **提交信息**: 
  ```
  feat: Implement VAD-based streaming ASR pipeline with speaker verification and transcription
  
  - Added VADStreamingOverlap3Pipeline for voice activity detection and audio segmentation.
  - Integrated silero_vad for natural speech boundary detection.
  - Implemented 3-source separation for detected speech segments.
  - Added speaker verification to filter target speakers using sherpa_onnx.
  - Enhanced ASR capabilities with real-time transcription and intermediate result handling.
  - Created StreamingASR class for managing audio streams and recognizing speech in real-time.
  - Introduced VADStreamingASR to combine VAD with streaming ASR for improved accuracy and responsiveness.
  ```

### 3. .gitignore 配置问题

当前 `.gitignore` 包含以下规则：
```
.idea
dataset/
models/
__pycache__/
*.onnx
*.pt
*.pth
*.ckpt
*.bin
*.wav
*.flac
*.mp3
segment-algo/test_results/
test/
cache/
todo*
test*/
other/
scripts/stream/
demo*
```

**问题**: `core.1021870` **未在 `.gitignore` 中**，因此被意外提交。

### 4. 仓库整体大小分析

| 项目 | 大小 |
|-----|------|
| `.git` 目录 | 808 MB |
| `core.1021870` | 3.6 GB |
| `dataset/` | 16 GB |
| `models/` | 1.3 GB |
| 其他文件 | ~100 MB |
| **总计** | **~21 GB** |

---

## ⚠️ 问题分析

### 问题 1: 不应该被追踪的大文件
- `core.1021870` 是一个 **3.6 GB 的核心转储文件**（通常是调试产生的临时文件）
- 不应该提交到版本控制系统
- 违反了版本控制最佳实践

### 问题 2: .gitignore 规则不完善
- 缺少针对 `core.*` 核心转储文件的规则
- 缺少针对其他临时文件的规则（如 `.dmp`, `.crash`, `.log` 等）

### 问题 3: 仓库膨胀
- git 仓库的 808 MB 大小中，包含了不必要的大文件
- 这会增加：
  - 克隆时间
  - 拉取/推送时间
  - 本地存储占用
  - 备份和维护成本

---

## 🛠️ 推荐解决方案

### 短期解决方案：将 core.1021870 添加到 .gitignore

```bash
echo "# Core dumps" >> .gitignore
echo "core.*" >> .gitignore
echo "*.crash" >> .gitignore
echo "*.dmp" >> .gitignore
```

这只会阻止**未来**的 `core.*` 文件被提交，但**不会删除**已提交的历史版本。

### 中期解决方案（推荐）：从 git 历史中清除大文件

使用 `git-filter-repo` 工具从整个历史中移除该文件：

```bash
# 1. 安装 git-filter-repo（如果未安装）
pip install git-filter-repo

# 2. 从历史中删除大文件
git filter-repo --path core.1021870 --invert-paths

# 3. 强制推送到远程（注意：会影响所有协作者）
git push origin --force-with-lease
```

**注意**: 这需要所有协作者重新克隆仓库。

### 长期解决方案：使用 Git LFS（Git Large File Storage）

对于大的二进制文件（如模型、数据集），考虑使用 Git LFS：

```bash
# 1. 安装 Git LFS
git lfs install

# 2. 追踪大文件类型
git lfs track "*.onnx"
git lfs track "*.pt"
git lfs track "*.wav"
git lfs track "core.*"

# 3. 提交 .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for large files"
```

---

## 📋 改进的 .gitignore 建议

```gitignore
# IDEs
.idea/
.vscode/
*.swp
*.swo

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.coverage

# Large files and models
dataset/
models/
cache/

# Binary files
*.onnx
*.pt
*.pth
*.ckpt
*.bin
*.wav
*.flac
*.mp3

# Core dumps and crash files
core.*
*.crash
*.dmp
*.dump

# Test and demo outputs
test/
test*/
test_*/
segment-algo/test_results/
test_overlap/
test_overlap_eval/
test-mossformer/
demo*/

# Build and temp
other/
*.log
*.tmp
*.temp
build/
dist/

# Specific script directories
scripts/stream/
scripts/demo/

# Documentation exclusions
todo*
```

---

## 📈 检查清单

- [ ] 立即将 `core.*` 添加到 .gitignore
- [ ] 从 git 历史中删除 `core.1021870`（如果想要减小仓库大小）
- [ ] 更新 .gitignore 以包含其他临时文件
- [ ] 考虑为大文件（>100MB）使用 Git LFS
- [ ] 通知所有协作者重新克隆仓库（如果进行了历史清理）
- [ ] 定期审查 git 仓库大小

---

## 🔗 相关资源

- [Git Large File Storage (LFS)](https://git-lfs.github.com/)
- [git-filter-repo 文档](https://github.com/newren/git-filter-repo)
- [gitignore 最佳实践](https://git-scm.com/docs/gitignore)

