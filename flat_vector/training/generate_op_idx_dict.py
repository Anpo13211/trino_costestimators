"""
スクリプト: op_idx_dict.json を生成する
使い方:
  python generate_op_idx_dict.py <root_plan_dir> -o op_idx_dict.json

指定したディレクトリ以下を再帰的に探索し、すべてのサブディレクトリにあるプランJSONから演算子名を収集し、
読み込んだJSONファイル名一覧も表示します（拡張子付きファイル名のみ）。
"""
import os
import json
import argparse


def collect_ops_recursively(root_dir, ext='.json'):
    """ディレクトリ以下を再帰的に探索し、すべてのプランJSONから演算子名を収集し、
    読み込んだファイルパスを返す"""
    ops = set()
    files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 'validation' ディレクトリは探索除外
        if 'validation' in dirnames:
            dirnames.remove('validation')

        for fname in filenames:
            if not fname.endswith(ext):
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 読み込んだファイルを記録
                files.append(path)
            except Exception as e:
                print(f"警告: {path} の読み込み失敗: {e}")
                continue

            # valid_queries ラッパー対応
            if isinstance(data, dict) and 'valid_queries' in data:
                for entry in data['valid_queries']:
                    plan = entry.get('plan')
                    if plan is not None:
                        recurse(plan, ops)
                continue

            # それ以外も再帰的に探索
            recurse(data, ops)

    return ops, files


def recurse(node, ops):
    """再帰的に dict/list をたどって演算子名を ops セットに追加"""
    if isinstance(node, dict):
        # 数字キーのみのラッパー(dict)を展開
        if node.keys() and all(isinstance(k, str) and k.isdigit() for k in node.keys()):
            for v in node.values():
                recurse(v, ops)
            return

        op = (
            node.get('name') or
            node.get('plan_parameters', {}).get('op_name') or
            node.get('nodeType') or         # Presto/Trino
            node.get('Node Type')           # PostgreSQL
        )
        if op:
            ops.add(op)

        # 子ノードを列挙
        children = []
        children += node.get('children', [])   # Presto/Trino
        children += node.get('Plans', [])      # PostgreSQL
        if 'plan' in node and isinstance(node['plan'], (dict, list)):
            children.append(node['plan'])

        for c in children:
            recurse(c, ops)

    elif isinstance(node, list):
        for elem in node:
            recurse(elem, ops)
    # それ以外は無視


def main():
    parser = argparse.ArgumentParser(description='演算子名→インデックス辞書を生成')
    parser.add_argument('root_dir', help='プランJSONルートディレクトリ（再帰探索）')
    parser.add_argument('-o', '--output', default='op_idx_dict.json', help='出力ファイル名')
    args = parser.parse_args()

    print(f"Scanning recursively in: {args.root_dir}")
    all_ops, files = collect_ops_recursively(args.root_dir)

    # 演算子インデックス辞書の生成
    op_list = sorted(all_ops)
    op_idx_dict = {op: idx for idx, op in enumerate(op_list)}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(op_idx_dict, f, indent=2, ensure_ascii=False)
    print(f"演算子数: {len(op_list)} を '{args.output}' に書き出しました 🎉")

    # 読み込んだJSONファイル名一覧を表示
    print("\n== 読み込んだJSONファイル名一覧 ==")
    for path in files:
        print(os.path.basename(path))

if __name__ == '__main__':
    main()
