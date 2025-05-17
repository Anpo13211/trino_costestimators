# trino_cost_estimators

Include the cost estimation models for Trino that I implemented.
Use the benchmark of [zero-shot](https://www.vldb.org/pvldb/vol15/p2361-hilprecht.pdf) to Train, Evaluate the models

## Prerequisites

- Python 3.8+  
- `venv` モジュール  
- 依存ライブラリは `requirements.txt` を参照

## Quick Start

```bash
# 仮想環境の作成と有効化
python -m venv env
source env/bin/activate

# パッケージ管理ツールのアップグレード
pip install --upgrade setuptools pip

# 依存パッケージのインストール
pip install -r requirements.txt
```

## Usage
```bash
# データセットの演算子を抽出
cd training
python generate_op_idx_dict.py <root_plan_dir> -o op_idx_dict.json

# 抽出成功後、モデルディレクトリに移動し、モデルを訓練する
cd model
python flat_vector.py
```

実行すると、以下の評価指標が出力されます:

- Q-error (50th, 90th percentile)

- MAPE

- RMSE

## Benchmark Extension
Zero-shot ベンチマークを使用してデータを増やしたり異なるワークロードのデータを集めたりできます:
https://github.com/DataManagementLab/zero-shot-cost-estimation

## Future Work
- [DACE](https://zheng-kai.com/paper/icde_2024_liang.pdf) や [T3](https://db.in.tum.de/~rieger/papers/t3.pdf) などの最新モデルの実装

- ハイパーパラメータチューニング機能の追加

- 大規模データセット対応の高速化
