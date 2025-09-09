#!/bin/bash

set -e
# 生成 speaker.txt，格式: speaker_name 绝对路径
find ../dataset/test -type f -name '*.wav' -print | sort | awk -F'/' '{speaker=$3; cmd="realpath "$0; cmd | getline abs; close(cmd); print speaker" "abs}' > ../dataset/speaker.txt

# 统计行数并展示前10行和后5行
wc -l ../dataset/speaker.txt
head -n 10 ../dataset/speaker.txt
printf '...\n'
tail -n 5 ../dataset/speaker.txt
