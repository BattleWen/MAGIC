#!/bin/bash
# 脚本名：download_datasets.sh
# 说明：关闭GPU后运行此脚本来预先下载所有数据集

cd /mnt/shared-storage-user/wenxiaoyu/game-private/eval/safety-eval-fork

echo "=========================================="
echo "开始下载通用能力基准数据集"
echo "=========================================="

# 1. 下载 MMLU
echo ""
echo "[1/5] 下载 MMLU 数据集..."
cd evaluation/tasks/generation/mmlu
wget -P . https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/eval
tar -xvf data.tar -C data
mv data/data data/eval/mmlu
rm -f data.tar
echo "✅ MMLU 下载完成"

# 2. 下载 GSM8K
echo ""
echo "[2/5] 下载 GSM8K 数据集..."
cd ../gsm8k
wget -P . https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl
echo "✅ GSM8K 下载完成"

# 3. 下载 BBH (Big Bench Hard)
echo ""
echo "[3/5] 下载 BBH 数据集..."
cd ../bbh
mkdir -p ../../../data/downloads/bbh
wget https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip -O ../../../data/downloads/bbh_data.zip
unzip ../../../data/downloads/bbh_data.zip -d ../../../data/downloads/bbh
mv ../../../data/downloads/bbh/BIG-Bench-Hard-main/* .
rm -rf ../../../data/downloads/bbh ../../../data/downloads/bbh_data.zip
echo "✅ BBH 下载完成"

# 4. 下载 Codex-Eval
echo ""
echo "[4/5] 下载 Codex-Eval 数据集..."
cd ../codex_eval
wget -P . https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
wget -P . https://huggingface.co/datasets/bigcode/humanevalpack/raw/main/data/python/data/humanevalpack.jsonl
echo "✅ Codex-Eval 下载完成"

# 5. 下载 TruthfulQA
echo ""
echo "[5/5] 下载 TruthfulQA 数据集..."
cd ../truthfulqa
mkdir -p data/eval/truthfulqa
wget -P data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv
echo "✅ TruthfulQA 下载完成"

echo ""
echo "=========================================="
echo "✓ 所有数据集下载完成！"
echo "=========================================="