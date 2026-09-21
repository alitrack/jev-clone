#!/usr/bin/env python3
"""把翻译的**原始输出**收敛成 v2 终稿：按 specs/M4.md §S3 整条剔除不合用的题。

为什么要单独一步（而不是让翻译脚本一次到位）：翻译跑一次 ~1 小时，而「哪些题不合用」的判据
在 M4-1 里改过三轮（先是不认千分位、再不认 `wan`/`1.8 million`、再是欧式小数逗号与 12 小时制）。
定稿必须能从**原始输出**重放，不能重跑翻译。规则与冻结尾门 `check-en-parity.py` 同源
（数值谓词来自 `scripts/qc_common.py`），跑完必须再过一次尾门才算数。

用法：finalize-mt.py <zh.jsonl> <raw_en.jsonl> <raw_report.json> <out_en.jsonl> <out_report.json>
"""
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qc_common import degenerate_reason, facts_lost  # noqa: E402


def read(path):
    return [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def key_of(item):
    return item["id"].split("-")[-1]


def slot_index(item):
    return sorted(item["question"]["criteria"]).index(item["gold"])


def unusable(zh, en):
    """返回剔除理由（None = 可用）。判据与尾门同源，只看**模型能看到的字段**。"""
    if zh["question"]["type"] == "choice" and slot_index(zh) != slot_index(en):
        return f"槽位漂移：{chr(65 + slot_index(zh))} -> {chr(65 + slot_index(en))}"
    lost = facts_lost(zh["state"], en["state"])
    lost += facts_lost(zh["question"].get("instructions") or "",
                       en["question"].get("instructions") or "")
    if zh["question"]["type"] == "choice":
        crit = json.dumps(en["question"]["criteria"], ensure_ascii=False)
        lost += [k for k in zh["question"]["criteria"] if facts_lost(k, crit)]
    if lost:
        return f"事实漂移：{sorted(set(lost))}"
    why = degenerate_reason(zh["state"], en["state"])
    if why:
        return f"译文损坏：{why}"
    return None


if __name__ == "__main__":
    zh_items, raw = read(sys.argv[1]), read(sys.argv[2])
    rep = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    zh_by = {key_of(i): i for i in zh_items}
    raw_by = {key_of(i): i for i in raw}

    drops = []          # (id, category, reason)
    for k in sorted(zh_by):
        if k not in raw_by:
            drops.append((k, zh_by[k]["category"], "译者已剔（槽位漂移/退化）"))
            continue
        why = unusable(zh_by[k], raw_by[k])
        if why:
            drops.append((k, zh_by[k]["category"], why))

    keep = [i for i in raw if key_of(i) not in {d[0] for d in drops}]
    out_en = pathlib.Path(sys.argv[4])
    out_en.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in keep) + "\n",
                      encoding="utf-8")
    sha = hashlib.sha256(out_en.read_bytes()).hexdigest()

    prov = [(k, facts_lost(zh_by[k].get("provenance") or "", raw_by[k].get("provenance") or ""))
            for k in sorted(set(zh_by) & set(raw_by)) if k not in {d[0] for d in drops}]
    prov = [(k, v) for k, v in prov if v]

    rep["dropped"] = [f"zh-ev-{k}({c}): {r}" for k, c, r in drops]
    rep["n_dropped"] = len(drops)
    rep["n_out"] = len(keep)
    rep["sha256"] = sha
    rep["dropped_by_category"] = dict(collections.Counter(c for _, c, _ in drops))
    rep["prov_drift_nonfatal"] = [f"zh-ev-{k}: {v}" for k, v in prov]
    pathlib.Path(sys.argv[5]).write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    print(f"终稿写出 {out_en}：{len(keep)} 条（原始 {len(raw)}，剔 {len(drops)}）")
    print(f"剔除分层：{rep['dropped_by_category']}")
    for k, c, r in drops:
        print(f"  - {k}({c}) {r[:90]}")
    print(f"provenance 非致命漂移 {len(prov)} 条（单列披露）：{[k for k, _ in prov]}")
    print(f"新 sha256 {sha}")
