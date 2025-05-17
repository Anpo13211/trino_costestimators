import os
import json
import pandas as pd
import numpy as np

# -----------------------------------------
# 事前準備：演算子名→インデックス辞書の読み込み
# -----------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_IDX_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "op_idx_dict.json"))
with open(OP_IDX_FILE, "r") as f:
    op_idx_dict = json.load(f)
print(f"Loaded {len(op_idx_dict)} operators from {OP_IDX_FILE}")
# ソートして固定順序のリストを作成
op_list = sorted(op_idx_dict.keys())
no_ops = len(op_list)

# -----------------------------------------
# 1) 固定長 flat vector を抽出する関数
# -----------------------------------------
def extract_flat_vector(plan_json):
    """
    各演算子タイプの出現回数と outputRowCount の合計を
    2*no_ops の固定長ベクトルで返す
    """
    num_vec  = np.zeros(no_ops, dtype=float)
    card_vec = np.zeros(no_ops, dtype=float)

    def recurse(node):
        op = node.get('name', 'Unknown')
        idx = op_idx_dict.get(op)
        if idx is not None:
            # 出現回数をインクリメント
            num_vec[idx] += 1
            # ノード直下の outputRowCount を加算
            if 'outputRowCount' in node:
                try:
                    card_vec[idx] += float(node['outputRowCount'])
                except:
                    pass
            # estimates 内の出力行数も加算
            for est in node.get('estimates', []):
                try:
                    card_vec[idx] += float(est.get('outputRowCount', 0))
                except:
                    pass
        # 子ノードを再帰処理
        for child in node.get('children', []):
            recurse(child)

    # plan_json のすべてのフラグメント(root)を処理
    for frag, root in plan_json.items():
        recurse(root)

    # 出現数 + カード合計 を連結して返却
    return np.concatenate([num_vec, card_vec])

# -----------------------------------------
# 2) 複数データセットの特徴量 DataFrame を生成
# -----------------------------------------
def build_multi_dataset_df(dataset_configs):
    """
    dataset_configs: [
      {"name":"tpch","plan_dir":"...","labels_csv":"..."},
      {"name":"accidents","result_file":"..."},
      ...
    ]
    各クエリプランから固定長 flat vector + メタ情報をまとめた DataFrame を返す
    """
    feats_list = []
    metas      = []

    for cfg in dataset_configs:
        name = cfg['name']

        # -- (A) plan_dir + labels_csv モード --
        if 'plan_dir' in cfg and 'labels_csv' in cfg:
            labels = pd.read_csv(cfg['labels_csv'])
            labels['key'] = labels['filename'].str.replace(r'\.sql$', '', regex=True)
            labels = labels[['key', 'wall_time_secs']].rename(columns={'wall_time_secs': 'runtime'})

            for fname in sorted(os.listdir(cfg['plan_dir'])):
                if not fname.endswith('.json'):
                    continue
                key = fname.split('_', 1)[0]
                plan_json = json.load(open(os.path.join(cfg['plan_dir'], fname), 'r'))
                vec = extract_flat_vector(plan_json)
                feats_list.append(vec)
                metas.append({
                    'runtime':    float(labels.loc[labels['key']==key, 'runtime'].values[0]),
                    'dataset_id': name
                })

        # -- (B) 単一 JSON ファイルモード --
        elif 'result_file' in cfg:
            data = json.load(open(cfg['result_file'], 'r'))
            for q in data.get('valid_queries', []):
                vec = extract_flat_vector(q['plan'])
                feats_list.append(vec)
                metas.append({
                    'runtime':    float(q.get('runtime_ms', 0)) / 1000.0,
                    'file':       q.get('file'),
                    'stmt_no':    q.get('stmt_no'),
                    'dataset_id': name
                })

        else:
            raise ValueError(f"Config for '{name}' must include plan_dir+labels_csv or result_file")

    # NumPy 配列に変換
    X = np.stack(feats_list, axis=0)  # shape = [n_queries, 2*no_ops]
    meta_df = pd.DataFrame(metas)

    # カラム名を作成
    count_cols = [f"{op}_count"   for op in op_list]
    card_cols  = [f"{op}_cardSum" for op in op_list]
    df_feats   = pd.DataFrame(X, columns=count_cols + card_cols)

    # 特徴量 + メタ情報を結合して返却
    df = pd.concat([df_feats, meta_df], axis=1)
    df = df[df['runtime'] > 0.0].reset_index(drop=True)
    return df
