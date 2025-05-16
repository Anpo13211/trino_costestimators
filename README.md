# trino_cost_estimators

Include the cost estimation models for Trino that I implemented.

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
# モデルディレクトリに移動
cd model

# flat_vector モデルの訓練とテストを実行
python flat_vector.py
```

実行すると、以下の評価指標が出力されます:

- Q-error (50th, 90th percentile)

- MAPE

- RMSE

## Benchmark Extension
Zero-shot ベンチマークを使用して他のデータセットやワークロードに拡張できます:
https://github.com/DataManagementLab/zero-shot-cost-estimation

## Future Work
- DACE や T3 などの最新モデルの実装

- ハイパーパラメータチューニング機能の追加

- 大規模データセット対応の高速化
