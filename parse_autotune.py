import re
import sys
from collections import Counter
from pathlib import Path

# ---------- 1. 正则表达式 ----------
# 匹配形如：
#   best config selected: BLOCK_M: 32, BLOCK_N: 128, num_warps: 8, ...
PATTERN = re.compile(
    r"""
    best\ config\ selected:\s*
    (?P<params>
        (?:\w+:\s*\w+(?:\.\w+)?,\s*)*
    )
    """,
    re.VERBOSE,
)

# ---------- 2. 解析函数 ----------
def parse_log(path: Path):
    """返回 (BLOCK_M, BLOCK_N, num_warps, num_stages) 的列表"""
    records = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            # 把 "BLOCK_M: 32, BLOCK_N: 128, ..." 变成 dict
            kv_str = m.group("params")
            kv_pairs = re.findall(r"(\w+):\s*([\w.]+)", kv_str)
            cfg = {k: v for k, v in kv_pairs}
            try:
                records.append(
                    (
                        int(cfg["BLOCK_M"]),
                        int(cfg["BLOCK_N"]),
                        int(cfg["num_warps"]),
                        int(cfg["num_stages"]),
                    )
                )
            except KeyError:
                continue
    return records

# ---------- 3. 生成 configs ----------
def make_configs(records):
    """
    records: [(BLOCK_M, BLOCK_N, num_warps, num_stages), ...]
    返回按指定顺序排序后的字符串列表
    """
    if not records:
        return []
    # 1. 四元组去重
    unique = sorted(set(records))  # 默认先按 M→N→warps→stages 升序
    configs = []
    for m, n, w, s in unique:
        configs.append(
            f"    triton.Config({{'BLOCK_M': {m}, 'BLOCK_N': {n}}}, "
            f"num_stages={s}, num_warps={w}),"
        )
    return configs

# ---------- 4. 主程序 ----------
def main():
    if len(sys.argv) != 2:
        print("Usage: python parse_autotune.py <autotune.log>", file=sys.stderr)
        sys.exit(1)

    log_path = Path(sys.argv[1])
    records = parse_log(log_path)
    cfgs = make_configs(records)

    print("configs = [")
    for line in cfgs:
        print(line)
    print("]")

if __name__ == "__main__":
    main()
