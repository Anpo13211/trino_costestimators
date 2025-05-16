import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# このファイル(__file__) の親フォルダ(= model/)のさらに上をパスに追加
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from training.extract_feature_flat import build_multi_dataset_df
from training.metrics import QError, MAPE, RMSE



def main():
    # ----------------------------------------------------------------------------
    # 1. データセット設定
    # ----------------------------------------------------------------------------
    dataset_configs = [
        {"name": "accidents",        "result_file": "../data_retrieve/test_datasets/accidents/accidents_valid.json"},
        {"name": "airline",          "result_file": "../data_retrieve/test_datasets/airline/airline_valid.json"},
        {"name": "baseball",         "result_file": "../data_retrieve/test_datasets/baseball/baseball_valid.json"},
        {"name": "basketball",       "result_file": "../data_retrieve/test_datasets/basketball/basketball_valid.json"},
        {"name": "carcinogenesis",   "result_file": "../data_retrieve/test_datasets/carcinogenesis/carcinogenesis_valid.json"},
        {"name": "consumer",         "result_file": "../data_retrieve/test_datasets/consumer/consumer_valid.json"},
        {"name": "credit",           "result_file": "../data_retrieve/test_datasets/credit/credit_valid.json"},
        {"name": "employee",         "result_file": "../data_retrieve/test_datasets/employee/employee_valid.json"},
        {"name": "fhnk",             "result_file": "../data_retrieve/test_datasets/fhnk/fhnk_valid.json"},
        {"name": "financial",        "result_file": "../data_retrieve/test_datasets/financial/financial_valid.json"},
        {"name": "geneea",           "result_file": "../data_retrieve/test_datasets/geneea/geneea_valid.json"},
        {"name": "genome",           "result_file": "../data_retrieve/test_datasets/genome/genome_valid.json"},
        {"name": "hepatitis",        "result_file": "../data_retrieve/test_datasets/hepatitis/hepatitis_valid.json"},
        {"name": "imdb",             "result_file": "../data_retrieve/test_datasets/imdb/imdb_valid.json"},
        {"name": "imdb_full",        "result_file": "../data_retrieve/test_datasets/imdb_full/imdb_full_valid.json"},
        {"name": "movielens",        "result_file": "../data_retrieve/test_datasets/movielens/movielens_valid.json"},
        {"name": "seznam",           "result_file": "../data_retrieve/test_datasets/seznam/seznam_valid.json"},
        {"name": "ssb",              "result_file": "../data_retrieve/test_datasets/ssb/ssb_valid.json"},
        {"name": "tournament",       "result_file": "../data_retrieve/test_datasets/tournament/tournament_valid.json"},
        {"name": "tpc_h",            "result_file": "../data_retrieve/test_datasets/tpc_h/tpc_h_valid.json"},
        {"name": "walmart",          "result_file": "../data_retrieve/test_datasets/walmart/walmart_valid.json"},
    ]

    # ----------------------------------------------------------------------------
    # 2. モデル保存ディレクトリ
    # ----------------------------------------------------------------------------
    validation_dir = "../data_retrieve/test_datasets/validation/"
    model_dir = "trained_model"
    os.makedirs(model_dir, exist_ok=True)

    # ----------------------------------------------------------------------------
    # 3. 評価指標 オブジェクト
    # ----------------------------------------------------------------------------
    metrics_objs = [
        QError(percentile=50, metric_prefix='test_'),
        QError(percentile=90, metric_prefix='test_'),
        RMSE(metric_prefix='test_'),
        MAPE(metric_prefix='test_'),
    ]

    results = []

    for test_cfg in dataset_configs:
        test_name = test_cfg['name']
        print(f"=== Testing on dataset: {test_name} ===")

        # 訓練用とテスト用の設定を分割
        train_cfgs = [cfg for cfg in dataset_configs if cfg['name'] != test_name]
        valid_cfgs = [
            {
                "name": cfg["name"],
                "result_file": os.path.join(
                    validation_dir,
                    cfg["name"],                   
                    f"{cfg['name']}_valid.json"
                )
            }
            for cfg in train_cfgs
        ]
        df_train = build_multi_dataset_df(train_cfgs)
        df_valid = build_multi_dataset_df(valid_cfgs)
        df_test  = build_multi_dataset_df([test_cfg])

        # 特徴量とラベルに分離
        meta_cols = ["file", "stmt_no", "dataset_id"]
        X_train = df_train.drop(columns=["runtime"] + meta_cols)
        y_train = df_train["runtime"].to_numpy()
        X_valid = df_valid.drop(columns=["runtime"] + meta_cols)
        y_valid = df_valid["runtime"].to_numpy()
        X_test  = df_test.drop(columns=["runtime"] + meta_cols)
        y_test  = df_test["runtime"].to_numpy()

        # 学習時に使われた特徴リストを取得し、テストにも同じ列を用意
        all_cols = (
            X_train.columns
            .union(X_valid.columns)
            .union(X_test.columns)
        )

        # ② その「全集合」で全 DataFrame をリインデックス＆ゼロ埋め
        X_train = X_train.reindex(columns=all_cols, fill_value=0)
        X_valid = X_valid.reindex(columns=all_cols, fill_value=0)
        X_test  = X_test.reindex(columns=all_cols, fill_value=0)

        # ----------------------------------------------------------------------------
        # 4. モデル訓練
        # ----------------------------------------------------------------------------
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbose': -1,
            'bagging_seed': 0,
        }
        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid,  label=y_valid, reference=train_set)

        callbacks = [
            lgb.callback.early_stopping(100),
            lgb.callback.log_evaluation(100)
        ]

        bst = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=['train','valid'],
            callbacks=callbacks
        )

        # ----------------------------------------------------------------------------
        # 5. モデル保存
        # ----------------------------------------------------------------------------
        model_path = os.path.join(model_dir, f"lgbm_l1o_{test_name}.txt")
        bst.save_model(model_path)
        print(f"Saved model : {model_path}")

        # ----------------------------------------------------------------------------
        # 6. 評価 (Metricクラス使用)
        # ----------------------------------------------------------------------------
        y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
        metrics_dict = {'test_datasets': test_name}
        for m in metrics_objs:
            m.evaluate(model=bst, metrics_dict=metrics_dict,
                       labels=y_test, preds=y_pred)
        print(",\ ".join([f"{k}={v:.4f}" for k,v in metrics_dict.items() if k!='test_datasets']))

        results.append(metrics_dict)

    # ----------------------------------------------------------------------------
    # 7. 結果まとめ
    # ----------------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    print("=== Summary of Leave-One-Out Results ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(model_dir, "l1o_summary.csv"), index=False)
    print(f"Summary saved to {os.path.join(model_dir, 'l1o_summary.csv')}")


if __name__ == '__main__':
    main()
