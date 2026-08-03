"""SpatialCLI RL 数据核验脚本(2026-08-03)。

对 IANNXANG/SpatialCLI 放出的 6350 条 RL 训练数据(parquet)做结构核验:
schema、行数、能力标注分布、奖励格式。用法:
  python inspect_rl_data.py <path-to-rl_train.parquet>
结论(已写入报告与线程 README):多选题格式,规则奖励押选项字母,
ability 列标注所需专家能力(如 'DP'=深度估计)。
"""

import sys

import pyarrow.parquet as pq


def main():
    t = pq.read_table(sys.argv[1])
    print("rows:", t.num_rows)
    print("schema:", t.schema)
    df = t.to_pandas()
    print("ability 分布:")
    print(df["ability"].value_counts().head(20))
    print("data_source 分布:")
    print(df["data_source"].value_counts().head(10))
    print("首条 reward_model:", df["reward_model"].iloc[0])
    print("首条 prompt 截选:", str(df["prompt"].iloc[0])[:500])


if __name__ == "__main__":
    main()
