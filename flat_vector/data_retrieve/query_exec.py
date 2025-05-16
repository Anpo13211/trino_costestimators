import os
import re
import argparse
import time
import json
from typing import List

from trino.dbapi import connect
from tqdm import tqdm

hostaddress = "localhost"

def clean_query(query: str) -> str:
    """コメントや余分なセミコロンを除去して整形"""
    lines: List[str] = []
    for line in query.splitlines():
        l = line.strip()
        if (not l or l.lower().startswith('set rowcount') or l.lower() == 'go'
                or l.startswith('--')):
            continue
        lines.append(l)
    return re.sub(r';\s*$', '', '\n'.join(lines))


_COL  = r'"[A-Za-z_][A-Za-z_0-9]*"\."?[A-Za-z_][A-Za-z_0-9]*"?'   # "tbl"."col"
_NUM  = r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)'                         # 123 12.3 .45
_CMP  = r'(=|!=|<>|<=|>=|<|>)'                                    # 比較演算子
_AGG  = r'\b(SUM|AVG|MIN|MAX)\b'                                  # 集約関数

def _to_double(col: str) -> str:
    """
    char(n)/varchar を安全に double に変える:
        TRY(CAST(TRIM(CAST(col AS varchar)) AS double))
    """
    return f'TRY(CAST(TRIM(CAST({col} AS varchar)) AS double))'

def fix_type_mismatch(sql: str) -> str:
    """
    1. 列 vs 数値の比較 ( = != < <= > >= )
       → 列を _to_double で包む
    2. SUM/AVG/MIN/MAX(列)
       → 引数を _to_double
    """

    q = sql

    # 1) 列 (比較) 数値   例: "tbl"."col" <= 123
    pat_col_num = re.compile(fr'({_COL})\s*{_CMP}\s*({_NUM})(?!\w)')
    q = pat_col_num.sub(lambda m: f'{_to_double(m.group(1))}{m.group(0)[len(m.group(1)) : ]}', q)

    # 2) 数値 (比較) 列   例: 999 = "tbl"."col"
    pat_num_col = re.compile(fr'({_NUM})\s*{_CMP}\s*({_COL})')
    q = pat_num_col.sub(lambda m: f'{m.group(1)}{m.group(0)[len(m.group(1)) : m.start(2)-m.start()]}{_to_double(m.group(2))}', q)

    # 3) 集約関数の引数を double 化
    pat_agg = re.compile(fr'{_AGG}\s*\(\s*({_COL})\s*\)', flags=re.I)
    q = pat_agg.sub(lambda m: f'{m.group(1).upper()}({_to_double(m.group(2))})', q)

    return q


def execute_workload(directory: str, *, catalog: str, schema: str, out_json: str,
                      max_valid: int = 5000, min_rows: int = 1,
                      min_runtime_ms: int = 50, timeout_sec: int = 50):
    """
    • directory   : 各行 1 SQL の .sql ファイルが置かれているディレクトリ
    • catalog / schema : Trino 接続先
    • out_json    : 収集したクエリ情報を書き出す JSON ファイル
    • max_valid   : 何本の「有効クエリ」を集めたら終了するか
    • min_rows    : 結果行数の下限
    • min_runtime_ms : 実行時間の下限
    • timeout_sec : サーバ & クライアント双方のタイムアウト
    """

    # -------- ① .sql ファイル一覧取得 --------------------------------------
    sql_files = sorted(f for f in os.listdir(directory) if f.endswith('.sql'))
    if not sql_files:
        raise FileNotFoundError('No .sql files in directory')

    valid = []
    processed = 0
    done = False
    bar = tqdm(total=max_valid, unit='qry', desc='Valid')

    # -------- ② ファイル単位で接続を張りっぱなしにする ----------------------
    for fname in sql_files:
        if done:
            break
        print(f'Processing {fname}…')

        # Trino に接続 (ファイル単位)
        conn = connect(host=hostaddress, port=8080, user='benchmark',
                       catalog=catalog, schema=schema,
                       session_properties={'query_max_run_time': f'{timeout_sec}s'})
        cur = conn.cursor()

        # ステートメント単位に分割
        path = os.path.join(directory, fname)
        with open(path, encoding='utf-8') as fh:
            raw_sql = fh.read()
        statements = [
            stmt.strip().rstrip(';')
            for stmt in raw_sql.split(';')
            if stmt.strip() and not stmt.strip().startswith('--')
        ]

        for stmt_no, q in enumerate(statements, start=1):
            if len(valid) >= max_valid:
                done = True
                break

            try:
                # デバッグ用出力 
                # print(f"Executing {fname} [stmt {stmt_no}]: {q}")

                # ----- EXPLAIN (FORMAT JSON) --------------------------------
                cur.execute(f'EXPLAIN (FORMAT JSON) {q}')
                plan_json_raw = cur.fetchone()[0]
                plan_json = json.loads(plan_json_raw)

                # ----- 実行 ---------------------------------------------------
                start_wall = time.monotonic()
                start_time = time.time()
                cur.execute(q)

                # while True:
                #     state = cur.stats.get('state')
                #     if state in ('FINISHED', 'FAILED', 'CANCELED'):
                #         break
                #     if time.monotonic() - start_wall > timeout_sec:
                #         cur.cancel()   # サーバーにキャンセル要求
                #         raise TimeoutError('client-side timeout')
                #     time.sleep(0.1)   # CPU を張り付きすぎないように

                # if cur.stats['state'] != 'FINISHED':
                #     raise RuntimeError('Query failed or cancelled')

                rows = cur.fetchall()
                runtime_ms = int((time.time() - start_time) * 1000)
                if len(rows) < min_rows or runtime_ms < min_runtime_ms:
                    raise RuntimeError('Did not meet row/time thresholds')

                # ----- 成功クエリを保存 ---------------------------------------
                stats = cur.stats
                valid.append({
                    'file': fname,
                    'stmt_no': stmt_no,
                    'sql': q,
                    'plan': plan_json,
                    'runtime_ms': runtime_ms,
                    'rows': len(rows),
                    'cpu_ms': stats.get('cpuTimeMillis', 0),
                    'peak_mem': stats.get('peakMemoryBytes', 0)
                })
                bar.update(1)

            except Exception as e:
                print(f'✗ 失敗 (processed={processed+1}): {e!r}')

            finally:
                processed += 1

        # ファイル単位でクローズ
        cur.close()
        conn.close()

        if len(valid) >= max_valid:
            done = True
            break

    bar.close()

    # -------- ③ 結果を書き出し ----------------------------------------------
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fp:
        json.dump({'catalog': catalog,
                   'schema': schema,
                   'valid_queries': valid},
                  fp, ensure_ascii=False, indent=2)
    print(f'Saved {len(valid)} valid queries → {out_json}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run workload on Trino, store plans + metrics')
    
    ap.add_argument('--workload_dir', 
                    default="../cross_db_benchmark/workloads/")
    ap.add_argument('--catalog',
                    default="accidents")
    ap.add_argument('--schema',
                    default='public')
    ap.add_argument('--output',
                    default='data_retrieve/')
    
    ap.add_argument('--max_valid',    type=int, default=50)
    ap.add_argument('--min_rows',     type=int, default=1)
    ap.add_argument('--min_runtime_ms', type=int, default=50)
    ap.add_argument('--timeout_sec',  type=int, default=30)
    ap.add_argument('--validation',   action='store_true', default=0, help='Use validation workload')
    args = ap.parse_args()

    # ★ ディレクトリ + ファイル名 を組み立てる
    workload_dir = os.path.join(args.workload_dir, args.catalog)
    if args.validation:
        workload_dir = os.path.abspath(workload_dir, "validation")
    if not os.path.exists(workload_dir):
        raise FileNotFoundError(f"Workload directory '{workload_dir}' does not exist")
    
    output = os.path.join(args.output, args.catalog)
    out_json = os.path.join(output, f'{args.catalog}_valid.json')

    execute_workload(
        workload_dir,
        catalog=args.catalog,
        schema=args.schema,
        out_json=out_json,             
        max_valid=args.max_valid,
        min_rows=args.min_rows,
        min_runtime_ms=args.min_runtime_ms,
        timeout_sec=args.timeout_sec
    )
