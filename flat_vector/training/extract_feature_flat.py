import os
import json
import pandas as pd
from collections import defaultdict


def extract_flat_vector(plan_json):
    """
    各演算子のインスタンス数（_*_count）と outputRowCount の合計（_*_cardSum）
    のみを抽出する（論文の flat vector 定義に沿う）:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
    """
    vec = defaultdict(float)

    def recurse(node):
        op = node.get('name', 'Unknown')
        vec[f"{op}_count"] += 1
        # ノード直下
        if "outputRowCount" in node:
            try:
                vec[f"{op}_cardSum"] += float(node["outputRowCount"])
            except:
                pass
        # estimates 内
        for est in node.get("estimates", []):
            try:
                vec[f"{op}_cardSum"] += float(est.get("outputRowCount", 0))
            except:
                pass
        # estimates 内の子ノード
        for child in node.get("children", []):
            recurse(child)

    for frag, node in plan_json.items():
        recurse(node)
    return dict(vec)

def build_multi_dataset_df(dataset_configs):
    """
    dataset_configs: [
        # plan_dir + labels_csv モード
        {"name": "tpch",   "plan_dir": ".../tpch/",   "labels_csv": ".../tpch.csv"},
        # 単一 JSON ファイルモード
        {"name": "accidents", "result_file": ".../accidents_valid.json"},
        # ... 他のデータセットも同様に
    ]
    上記いずれかの設定ごとに flat vector + メタ情報をまとめて返す
    """
    all_dicts = []

    for cfg in dataset_configs:
        name = cfg["name"]

        # ① 従来の plan_dir + labels_csv モード
        if "plan_dir" in cfg and "labels_csv" in cfg:
            labels = pd.read_csv(cfg["labels_csv"])
            
            # 列名を変更し、見やすくしておく
            labels["key"] = labels["filename"].str.replace(r"\.sql$", "", regex=True)
            labels = labels[["key", "wall_time_secs"]].rename(columns={"wall_time_secs":"runtime"})
            
            for fname in sorted(os.listdir(cfg["plan_dir"])):
                if not fname.endswith(".json"):
                    continue
                key = fname.split('_',1)[0]
                plan_json = json.load(open(os.path.join(cfg["plan_dir"], fname)))
                feats = extract_flat_vector(plan_json)
                feats["runtime"]    = labels.loc[labels["key"]==key, "runtime"].values[0]
                feats["dataset_id"] = name
                all_dicts.append(feats)

        # ② 単一 JSON ファイルモード
        elif "result_file" in cfg:
            data = json.load(open(cfg["result_file"], "r"))

            # 1つの JSON ファイルに複数のクエリが含まれているから、クエリごとに特徴を抽出する
            for q in data.get("valid_queries", []):
                feats = extract_flat_vector(q["plan"])
                # ラベル・メタ情報を追加
                feats.update({
                    "runtime":   q.get("runtime_ms", 0) / 1000.0,   # 秒に変換
                    # "cpu_ms":    q.get("cpu_ms", None),
                    # "peak_mem":  q.get("peak_mem", None),
                    "file":      q.get("file"),
                    "stmt_no":   q.get("stmt_no"),
                    "dataset_id": name,
                })
                all_dicts.append(feats)

        else:
            raise ValueError(f"Config for '{name}' must include either plan_dir+labels_csv or result_file")

    df = pd.DataFrame(all_dicts)
    df = df[df["runtime"] > 0.0].copy()  # runtime が 0 のものは除外
    df = df.fillna(0)  # NaN を 0 に置換
    return df