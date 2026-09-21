#!/usr/bin/env python3
"""M4-1 — 第二个独立翻译源：把 `zh-evidence-v1` 译成英文对照集（v2）。

为什么要专门的 MT 模型
----------------------
M3 的 `eval/items/en-evidence-v1.jsonl` 是**被评模型自己**（115:8014 的 qwen3.8-27b）译的，
译者与考生是同一个系统。这条混淆写在 M3 报告里，是「中英差异 → 中文能力」之间唯一的拦路石。
`specs/M4.md` §D1 要求第二译源**与被评模型不共享失败模式**且**可离线复现**。

本脚本用规格首选 `facebook/nllb-200-distilled-600M`（专用 MT 模型、非 LLM、纯 CPU、权重可钉哈希；
许可 CC-BY-NC-4.0 ⇒ **只本地评测用、不进对外产物**）。权重经**魔搭 ModelScope** 下载（用户指定优选国内源）：
`modelscope download --model facebook/nllb-200-distilled-600M --local_dir <dir>`。
transformers 5.17 已删除 NLLB 建模类（只剩 `models/nllb/tokenization_nllb.py`、无 `modeling_nllb.py`），
故 4.44.2 装在**隔离目录**并用 `PYTHONPATH` 前置（不动 venv-laya 的 5.17，保住 M3 的可复现性）：
`pip install --target <隔离目录> 'transformers==4.44.2'`。

四件会静默毁掉对照的事（脚本各自做了防护）
------------------------------------------
1. **目标语言标记**。NLLB 拿不到 `forced_bos_token_id` 时会**自己乱挑语种**——实测输出印地语、希腊语、
   意大利语、乌兹别克语并伴随重复循环，而整批数据看起来「翻译完成了」。脚本强制设定它，并跑**探针
   自检**（探针译文里非拉丁文字必须为 0）与**占位符存活自检**，任一不通过即中止。
2. **槽位序**。`choice` 题的槽位＝`criteria` 键**按字典序（码点序）**排好后的位置——`criteria` 是
   `BTreeMap`（`crates/jev-core/src/render.rs:141-142` 明写 "iteration is in dictionary order of the keys"），
   所以 **JSON 里的书写顺序不影响槽位，类型决定**。
   译文标签一翻序，正解就从 B 滑到 A，配对看着仍然正常，实际比的却是模型的槽位偏好而不是语言能力。
   故脚本逐条验算：MT 英文标签的码点序若 ≠ 中文槽位序（实测 `合规/违规`→"Compliance/Breaches"），
   该题**整条剔除**——S3 只允许剔除，**不许人工改译文**（人工干预＝新混淆）。MT 把短标签译错
   （实测 `否`→"I don't know."、`能得出`→"Can you find out?"、`丙某`→"丙 is one of them."）同样只能剔题，
   剔除逐条列名进 `n_missing`。每个标签的 MT 原文记进 `term_audit`，供复核复算。
3. **gold 可达性**。state 里写 "Option 1" 而选项栏写 "Option 一" 时，模型无法把证据映射到选项，
   **gold 变成不可达**——这不是翻译风格问题，是把题弄坏了。故文本里出现的标签先换占位符、译完展开成
   **选项栏那一批英文串**，两边用词由构造保证一致；state 里另有不属于任何标签的标识符指代（如「一号楼」）
   则单列进 `ident_mentions`。
4. **事实漂移 / 结构漂移**。数字与标识符 token 必须在英文里存活（不存活即列名，供抽检对照）；
   `category`、题型、选项数、`score` 顺序、`gold` 一律**照抄不重生成**。

用法：
  translate-mt.py --scan                        # 盘点：标签集 / 需保护的字符 / 分数档位
  translate-mt.py --scan-labels <zh.jsonl>      # 逐标签集打印 MT 首选 vs 采用值（是否翻序）
  translate-mt.py <zh.jsonl> <out.jsonl> [--report R] [--print-manifest-entry]
  运行须 PYTHONPATH=<transformers-4.44 隔离目录>
"""

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import re
import sys

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
NUM_RE = re.compile(r"\d+")
ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*")

# 需原样保留的标识符字符（M3 的「标识符铁律」：甲/乙/丙 在英文集里也保持原样）。
# 这些字在中文句子里**只会作为标识符**出现 ⇒ 长文本里也可以全量保护。
# `--scan` 会列出中文题集里实际出现的汉字及频次，用来核对这张表够不够。
IDENT_CHARS = "甲乙丙丁戊己庚辛壬癸"
# 中文数字只在**标签**里保护：标签里的「方案一/二号楼」是标识符，保住它才能保住码点序；
# 但长文本里的「一/二/三」是普通数词（「一体化」「第二条」），全量保护会把句子毁掉。
CJK_NUM = "一二三四五六七八九十"


def _ph(ch: str) -> str:
    if ch in IDENT_CHARS:
        return f"ZQ{IDENT_CHARS.index(ch) + 1}Z"
    return f"ZN{CJK_NUM.index(ch) + 1}Z"


def protect(text: str, chars: str = IDENT_CHARS) -> str:
    for ch in chars:
        text = text.replace(ch, _ph(ch))
    return text


def unprotect(text: str) -> str:
    def sub(m: re.Match) -> str:
        token = m.group(0).upper()
        for ch in IDENT_CHARS + CJK_NUM:
            if _ph(ch) == token:
                return ch
        return m.group(0)

    return re.sub(r"[Zz][QNqn]\d+[Zz]", sub, text)


def protect_label(label: str) -> str:
    """标签里的标识符（甲/乙/丙 + 中文数字）原样保留。"""
    return protect(label, IDENT_CHARS + CJK_NUM)


def is_ident_label(label: str) -> bool:
    """该标签是不是**标识符型**（决定它要不要在长文本里统一替换）。

    只有标识符型才需要：`Option 一` 与 `Option 1` 之间模型没法做语义映射，gold 会不可达。
    而 `支持` / `有权` 这类语义标签在散文里本来就是自然语言，硬替换会把句子改坏
    （实测把「买方有权解除合同」改成 "the buyer Rights granted cancelled the contract"）。
    """
    return any(ch in IDENT_CHARS or ch in CJK_NUM for ch in label)

# ── 标签处置：只做「标识符原样」，**不做人工改译文**（specs/M4.md §S3 ⛔）
#
# S3 写死：⛔ 禁止人工顺手改译文（人工干预＝新混淆）；单条不可用只允许**整条剔除**，
# 计入 n_missing 并逐条列名。所以这里**没有**「换人工同义词」的表——MT 译什么就是什么。
# 只保留两件**结构性**处理（不改变「谁译的」这个变量）：
#   ① 标识符原样（甲/乙/丙 + 标签里的 一/二/三）——M3 的标识符铁律，也是保住码点序的唯一办法；
#   ② 标识符型标签在长文本里统一替换成选项栏那一批英文串（否则 gold 不可达）。
#
# 标签可用不可用由人**判定**（S3 的抽检口径），判「不可用」的题**整条剔除**。两类判据：
#   (a) 机械判据：英文标签的码点序 ≠ 中文槽位序（槽位漂移会把位置偏好掺进配对）——脚本自动判；
#   (b) 语义判据：MT 把短标签译错（实测 `否`→"I don't know."、`能得出`→"Can you find out?"）——
#       人读一遍标签清单后写入下表（键＝该题的标签集元组，值＝不可用理由）。
# 判定结果与该 run 的引擎绑定：换引擎要重判（引擎输出不同，判据也不同）。
UNUSABLE_LABEL_SETS: dict[tuple[str, ...], str] = {}

# 候选译源注册表（specs/M4.md §D1 的主选 + 备选）。S3 要求：抽检不合用就换备选并记进报告。
# 两个引擎的**接口不同**：NLLB 靠 forced_bos_token_id=语言标记；MADLAD 是 T5 系，目标语言写成输入前缀。
ENGINES: dict[str, dict] = {
    "nllb600": {
        "id": "facebook/nllb-200-distilled-600M",
        "loader": "transformers 4.44.2 隔离目录 + AutoModelForSeq2SeqLM（多语 MT，靠 forced_bos_token_id 指定目标语言）",
        "params": "num_beams=4 forced_bos_token_id=eng_Latn src_lang=zho_Hans no_repeat_ngram_size=3 max_length=min(512, 3×源句长+40)",
        "license": "CC-BY-NC-4.0（非商用）⇒ 只本地评测用，不得进对外产物",
        "prefix": "", "src_lang": "zho_Hans", "tgt_lang": "eng_Latn",
    },
    "madlad3b": {
        "id": "google/madlad400-3b-mt",
        "loader": "transformers 4.44.2 隔离目录 + AutoModelForSeq2SeqLM（T5 系，目标语言写成输入前缀 <2en>）",
        "params": "num_beams=4 输入前缀 <2en> no_repeat_ngram_size=3 max_length=min(512, 3×源句长+40)",
        "license": "Apache-2.0（可商用、可进对外产物）",
        "prefix": "<2en> ", "src_lang": None, "tgt_lang": None,
    },
}


def read_items(path: pathlib.Path) -> list[dict]:
    items = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            items.append(json.loads(s))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from None
    return items


def split_sentences(text: str) -> list[str]:
    """中文按句末标点切句；MT 是句级模型，整段丢进去会更容易丢内容。"""
    parts = re.split(r"(?<=[。！？；])", text)
    return [p for p in (x.strip() for x in parts) if p]


def load_engine(engine: str, model_id: str, cache_dir: str, num_beams: int):
    """按引擎类型装载。多语 MT 与 T5 系（MADLAD）的接口不同，这里分开处理。"""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cfg = ENGINES[engine]
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=cache_dir)
    model.eval()
    print(f"[engine] {cfg['id']}｜类 {type(model).__name__}｜参数量 {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    forced_bos = None
    if cfg["tgt_lang"]:
        forced_bos = tok.convert_tokens_to_ids(cfg["tgt_lang"])
        if forced_bos in (None, tok.unk_token_id):
            raise SystemExit(f"译源 {cfg['id']} 不支持目标语言标记 {cfg['tgt_lang']!r}"
                             "（多语 MT 必须显式指定，否则会输出任意语种）")
        print(f"[engine] 目标语言 {cfg['tgt_lang']} -> forced_bos_token_id={forced_bos}", flush=True)
    if cfg["src_lang"] and hasattr(tok, "src_lang"):
        tok.src_lang = cfg["src_lang"]

    prefix = cfg["prefix"]

    def translate(text: str) -> str:
        out: list[str] = []
        for sent in split_sentences(text):
            enc = tok(prefix + sent, return_tensors="pt")
            # 输出上限按**源句长度**给（英文 token 数约为中文字符数的 1.3 倍），而不是一律 512：
            # 3B 模型在 2 字标签上会陷入复读机（实测 `否/是` → "No, no, no…"），一律 512 会让
            # beam search 跑满上限（单条 2–4 分钟，实测全量因此从 ~30 分钟劣化到 4 小时以上）。
            # `no_repeat_ngram_size=3` 是标准的防复读解码护栏，两个参数都写进报告，可复算。
            cap = max(64, min(512, 3 * len(sent) + 40))
            kw: dict = {"num_beams": num_beams, "max_length": cap, "no_repeat_ngram_size": 3}
            if forced_bos is not None:
                kw["forced_bos_token_id"] = forced_bos
            with torch.no_grad():
                gen = model.generate(**enc, **kw)
            out.append(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
        return " ".join(out)

    return tok, model, translate


NON_LATIN = re.compile(r"[\u4e00-\u9fff\u0900-\u097f\u0400-\u04ff\u0370-\u03ff\u0600-\u06ff\u0e00-\u0e7f\u10a0-\u10ff]")


def engine_healthcheck(translate) -> str:
    """探针：译源必须真的输出英文。多语模型的「语言标记没生效」是静默失败——
    它照样返回 200、照样是流畅句子，只是语言错了；没有这一步就发现不了。"""
    probe = translate("这是一条用来检查译源是否正常工作的中文测试句。")
    print(f"[engine] 探针译文：{probe!r}", flush=True)
    if not probe or NON_LATIN.search(probe):
        raise SystemExit(
            f"译源自检失败：探针译文 {probe!r} 不是英文（含非拉丁文字）。"
            "多半是目标语言标记没生效——检查 --tgt-lang。"
        )
    # 占位符必须原样存活，否则「标签统一替换」这条防线是假的（替换串被 MT 改掉就展不开）
    ph_probe = translate("本单与 QX1Q 的差异：QX2Q 与 QX3Q 相比更省时。")
    alive = sum(1 for t in ("QX1Q", "QX2Q", "QX3Q") if t in ph_probe.upper())
    print(f"[engine] 占位符探针：{ph_probe!r} → 存活 {alive}/3", flush=True)
    if alive < 3:
        raise SystemExit(f"占位符自检失败（{ph_probe!r}）：标签统一替换防线不可用，请换占位符写法。")
    return probe


def snapshot_dir(model_id: str, cache_dir: str) -> pathlib.Path | None:
    owner, _, name = model_id.partition("/")
    root = pathlib.Path(cache_dir) / f"models--{owner}--{name}"
    snaps = sorted((root / "snapshots").glob("*")) if (root / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def engine_ref(model_id: str, cache_dir: str) -> tuple[pathlib.Path | None, str]:
    """返回（权重所在目录，可直接写进报告的版本串）。

    支持三种来源：本地目录（ModelScope `--local_dir` 下载的形态）、HF 缓存、纯模型名。
    本机走的是**魔搭**：`modelscope download --model Helsinki-NLP/opus-mt-zh-en --local_dir …`，
    故这里首选「本地目录」分支；附带的 metadata.json 里记着魔搭仓的 revision。
    """
    p = pathlib.Path(model_id)
    if p.is_dir():
        rev = "local:" + p.name
        meta = p / "metadata.json"
        if meta.is_file():
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
                bits = [str(m.get(k)) for k in ("Revision", "CommitId", "Branch") if m.get(k)]
                if bits:
                    rev = f"modelscope:{'/'.join(bits)}"
            except (json.JSONDecodeError, OSError):
                pass
        return p, rev
    snap = snapshot_dir(model_id, cache_dir)
    return snap, (snap.name if snap else "unknown")


def weights_sha256(snap: pathlib.Path | None) -> str:
    if snap is None:
        return "unknown"
    for fname in ("model.safetensors", "pytorch_model.bin"):
        f = snap / fname
        if f.is_file():
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return f"{fname}:{h.hexdigest()}"
    return "unknown"


def mt_label(label: str, translate) -> str:
    """MT 对该标签的原始译法（标识符经占位符保护后原样还原）。"""
    return unprotect(translate(protect_label(label)))


def substitute_terms(text: str, term_map: dict[str, str]) -> tuple[str, dict[str, str]]:
    """把长文本里出现的**标识符型**标签换成占位符，译完再展开成选项栏那一批英文串。

    只处理标识符型：`Option 一` 与 `Option 1` 之间模型没法做语义映射，gold 会不可达；
    而 `支持`/`有权` 这类语义标签在散文里本来就是自然语言，硬替换会把句子改坏
    （实测把「买方有权解除合同」改成 "the buyer Rights granted cancelled the contract"）。
    """
    mapping: dict[str, str] = {}
    for i, (zh, en) in enumerate(sorted(term_map.items(), key=lambda kv: -len(kv[0])), 1):
        if zh and is_ident_label(zh) and zh in text:
            ph = f"QX{i}Q"
            text = text.replace(zh, ph)
            mapping[ph] = en
    return text, mapping


def expand_placeholders(text: str, mapping: dict[str, str]) -> str:
    for ph, en in mapping.items():
        text = re.sub(re.escape(ph), lambda _m, e=en: e, text, flags=re.IGNORECASE)
    return text


def choose_labels(zh_keys_sorted: list[str], translate, term_audit: dict[str, dict],
                  item_id: str) -> tuple[dict[str, str] | None, str | None]:
    """返回 `(标签映射, 不可用理由)`；理由非 None ⇒ 该题**整条剔除**（specs/M4.md §S3）。

    判据：(a) 机械——MT 英文标签的码点序必须等于中文槽位序（否则槽位漂移）；
    (b) 人工——`UNUSABLE_LABEL_SETS` 里登记的语义错。
    **不做**「换成人工同义词」：S3 禁止人工改译文。
    """
    mapping: dict[str, str] = {}
    for k in zh_keys_sorted:
        if k not in term_audit:
            term_audit[k] = {"mt": mt_label(k, translate)}
        mapping[k] = term_audit[k]["mt"]
    vals = [mapping[k] for k in zh_keys_sorted]
    if len(set(vals)) != len(vals):
        return None, f"英文标签重复（MT 把不同标签译成同一串）：{vals}"
    if vals != sorted(vals):
        return None, (f"槽位漂移：中文槽位序 {zh_keys_sorted} → MT 英文序 {vals}，"
                      f"英文排序后为 {sorted(vals)}")
    verdict = UNUSABLE_LABEL_SETS.get(tuple(zh_keys_sorted))
    if verdict:
        return None, f"人判不可用：{verdict}"
    return mapping, None


# 事实核对 / 退化检测的实现**搬到 scripts/qc_common.py**：冻结尾门 check-en-parity.py 也要
# 用同一个谓词（否则英文集落盘后就没人在验事实），单一实现避免两处各判一套。
# 这里只做导入，调用点（facts_lost / degenerate_reason / fact_tokens）保持不变。
from qc_common import fact_tokens, facts_lost, degenerate_reason, REPEAT_RE  # noqa: E402,F401


def scan(items: list[dict]) -> None:
    cjk = collections.Counter()
    for it in items:
        for ch in it["state"]:
            if CJK_RE.search(ch):
                cjk[ch] += 1
        for ch in it["question"].get("instructions", ""):
            if CJK_RE.search(ch):
                cjk[ch] += 1
    print("=== 中文题集里出现的汉字（频次前 40）===")
    print(" ".join(f"{c}:{n}" for c, n in cjk.most_common(40)))
    print(f"\n=== 需保护的候选字符是否都在题集里 ===")
    print(" ".join(f"{c}:{'✓' if c in cjk else '—'}" for c in IDENT_CHARS))

    choice_sets = collections.Counter()
    score_lists = collections.Counter()
    for it in items:
        q = it["question"]
        if q["type"] == "choice":
            keys = tuple(sorted(q["criteria"].keys()))
            choice_sets[keys] += 1
        elif q["type"] == "score":
            score_lists[tuple(q["criteria"])] += 1
    print(f"\n=== choice 标签集（{len(choice_sets)} 种，共 {sum(choice_sets.values())} 条）===")
    for keys, n in choice_sets.most_common():
        print(f"  n={n:3d}  {list(keys)}")
    print(f"\n=== score 档位列表（{len(score_lists)} 种）===")
    for keys, n in score_lists.most_common():
        print(f"  n={n:3d}  {list(keys)}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zh", nargs="?", type=pathlib.Path)
    ap.add_argument("out", nargs="?", type=pathlib.Path)
    ap.add_argument("--scan", action="store_true", help="只盘点，不翻译")
    ap.add_argument("--scan-labels", action="store_true",
                    help="加载译源后只译标签，逐集打印「中文槽位序 / MT 序 / 是否翻序」")
    ap.add_argument("--engine", default="madlad3b", choices=sorted(ENGINES),
                    help="译源引擎（specs/M4.md §D1：主选 nllb600，S3 不合用则换 madlad3b）")
    ap.add_argument("--model", default=None, help="本地权重目录或 HF/魔搭 id（默认取引擎注册表里的 id）")
    ap.add_argument("--cache-dir", default="/mnt/d/wsl2/tmp/jev-m4/hf-cache")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--source-tag", default="authored-en-v2-mt")
    ap.add_argument("--report", type=pathlib.Path, default=None)
    ap.add_argument("--print-manifest-entry", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只译前 N 条（试跑用）")
    args = ap.parse_args(argv[1:])
    if args.model is None:
        args.model = ENGINES[args.engine]["id"]

    if args.scan:
        if not args.zh:
            print("--scan 需要 <zh.jsonl>", file=sys.stderr)
            return 2
        scan(read_items(args.zh))
        return 0
    if args.scan_labels:
        _, _, translate = load_engine(args.engine, args.model, args.cache_dir, args.num_beams)
        engine_healthcheck(translate)
        sets: dict[tuple[str, ...], int] = {}
        for it in read_items(args.zh):
            q = it["question"]
            if q["type"] == "choice":
                keys = tuple(sorted(q["criteria"].keys()))
                sets[keys] = sets.get(keys, 0) + 1
        drift = []
        print("=== 逐标签集：中文槽位序 vs MT 首选序 ===")
        for keys, n in sorted(sets.items(), key=lambda kv: -kv[1]):
            mt = [unprotect(translate(protect(k))) for k in keys]
            ok = len(set(mt)) == len(mt) and list(mt) == sorted(mt)
            if not ok:
                drift.append((keys, mt))
            print(f"  n={n:3d} {'OK   ' if ok else 'DRIFT'} zh={list(keys)}")
            print(f"            mt={mt}")
            if not ok:
                print(f"            sorted={sorted(mt)}   ← 漂了 ⇒ 该题整条剔除（S3：不许人工改译文）")
        print(f"\n共 {len(sets)} 种标签集｜翻序 {len(drift)} 种")
        return 0

    if not (args.zh and args.out):
        print(__doc__)
        return 2

    zh_items = read_items(args.zh)
    if args.limit:
        zh_items = zh_items[: args.limit]

    tok, model, translate = load_engine(args.engine, args.model, args.cache_dir, args.num_beams)
    probe = engine_healthcheck(translate)

    fact_flags: list[str] = []
    out_items: list[dict] = []
    label_cache: dict[tuple[str, ...], tuple[dict[str, str] | None, str | None]] = {}
    term_audit: dict[str, dict] = {}      # 每个标签的 MT 译法（供复核）
    ident_mentions: list[str] = []        # state 里出现、但**不属于任何标签**的标识符类指代
    dropped: list[str] = []               # 整条剔除的题（S3：不可用只允许剔除，不许人工改）

    for n, z in enumerate(zh_items, 1):
        zq = z["question"]
        kind = zq["type"]

        if kind == "choice":
            keys = sorted(zq["criteria"].keys())
            cache_key = tuple(keys)
            if cache_key not in label_cache:
                label_cache[cache_key] = choose_labels(keys, translate, term_audit, z["id"])
            label_map, why = label_cache[cache_key]
            if why is not None or label_map is None:
                dropped.append(f"{z['id']}({z['category']}): {why}")
                continue
            criteria_en = {label_map[k]: None for k in keys}
            gold_en = label_map[z["gold"]]
        elif kind == "score":
            label_map = {}
            for lab in zq["criteria"]:
                if lab not in term_audit:
                    term_audit[lab] = {"mt": mt_label(lab, translate)}
                label_map[lab] = term_audit[lab]["mt"]
            criteria_en = [label_map[lab] for lab in zq["criteria"]]  # 顺序即档位，照抄
            bad = next((lab for lab in zq["criteria"] if (lab,) in UNUSABLE_LABEL_SETS), None)
            if len(set(criteria_en)) != len(criteria_en):
                dropped.append(f"{z['id']}({z['category']}): 英文档位标签重复 {criteria_en}")
                continue
            if bad:
                dropped.append(f"{z['id']}({z['category']}): 人判不可用：{UNUSABLE_LABEL_SETS[(bad,)]}")
                continue
            gold_en = z["gold"]                                      # score 的 gold 是档位索引
        else:  # noul：criteria/gold 由契约固定（true/false），不译
            label_map = {}
            criteria_en = zq.get("criteria")
            gold_en = z["gold"]

        def tr(text: str, _lm=label_map) -> str:
            """先按标签占位（保证与选项栏同一批英文串），再 MT 其余部分。"""
            subbed, mapping = substitute_terms(text, _lm)
            return expand_placeholders(unprotect(translate(protect(subbed))), mapping)

        # 一致性体检：state 里若出现「一号楼」这类标识符指代、却不属于任何标签，
        # 标签替换就管不到它 ⇒ 译文可能与选项栏用词不一致（gold 不可达）。单独记下来。
        for ref in re.findall(r"第?[一二三四五六七八九十]+[号楼幢期条项]", z["state"]):
            if not any(ref in lab for lab in label_map):
                ident_mentions.append(f"{z['id']}: state 里的「{ref}」不在任何标签内，未被统一替换")

        q_en: dict = {}
        for k, v in zq.items():
            if k == "type":
                q_en[k] = v
            elif k == "instructions":
                q_en[k] = tr(v)
            elif k == "criteria":
                q_en[k] = criteria_en
            else:
                q_en[k] = v

        en = {
            "id": "en-" + z["id"][len("zh-"):],
            "category": z["category"],
            "source": args.source_tag,
            "state": tr(z["state"]),
            "question": q_en,
            "gold": gold_en,
        }
        if "positive" in z:
            p = z["positive"]
            if kind == "choice":
                en["positive"] = label_map[p] if p in label_map else tr(p)
            elif kind == "score":
                en["positive"] = label_map[p] if p in label_map else tr(p)
            else:
                en["positive"] = p
        if "provenance" in z:
            en["provenance"] = tr(z["provenance"])

        # 机械退化检测（复读/长度爆炸）⇒ 整条剔除。测的是「译文坏了」，不是「译文不好」。
        labels_txt = (" ".join(str(v) for v in criteria_en.values()) if isinstance(criteria_en, dict)
                      else " ".join(str(v) for v in (criteria_en or [])))
        deg = next((f"{name}：{r}" for name, txt in (
            ("state", en["state"]),
            ("instructions", str(q_en.get("instructions", ""))),
            ("provenance", str(en.get("provenance", ""))),
            ("标签", labels_txt),
        ) if (r := degenerate_reason(z["state"], txt))), None)
        if deg:
            dropped.append(f"{z['id']}({z['category']}): 译文退化（{deg}）")
            continue

        lost = facts_lost(z["state"], en["state"])
        if lost:
            fact_flags.append(f"{z['id']}: 英文 state 丢掉 token {lost}")
        for ch in IDENT_CHARS:
            if ch in z["state"] and ch not in en["state"]:
                fact_flags.append(f"{z['id']}: 受保护标识符 {ch} 未存活")
        out_items.append(en)
        if n % 20 == 0:
            print(f"  已译 {n}/{len(zh_items)}", flush=True)

    snap, rev = engine_ref(args.model, args.cache_dir)
    sha = weights_sha256(snap)
    header = [
        "# en-evidence-v2 —— `zh-evidence-v1` 的**第二独立译源**英文对照集（M4-1）",
        "#",
        f"# 译源：{ENGINES[args.engine]['id']}（专用 MT 模型、非 LLM）｜本地目录 {rev}",
        f"# 权重 sha256：{sha}",
        f"# 加载：{ENGINES[args.engine]['loader']}｜运行须 PYTHONPATH=<transformers-4.44 隔离目录>",
        f"# 权重来源：魔搭 ModelScope（modelscope download --model {ENGINES[args.engine]['id']}）",
        f"# 生成参数：{ENGINES[args.engine]['params']}（脚本 scripts/translate-mt.py）",
        f"# 仪器自检：探针译文 {probe!r}",
        f"# 许可：{ENGINES[args.engine]['license']}",
        "#",
        "# 为什么换译源：M3 的 en-evidence-v1 由**被评模型自己**翻译（译者＝考生）。本集用专用 MT",
        "# 模型重译同一批中文原件，中文侧一字未动 ⇒ 唯一被换掉的变量是译者。见 specs/M4.md §D1/D2。",
        "#",
        "# 处置纪律（specs/M4.md §S3）：**禁止人工顺手改译文**；单条不可用只允许**整条剔除**，",
        "# 计入 n_missing 并逐条列名（见下方「整条剔除明细」）。故标签与长文本**一律是 MT 原文**，",
        "# 只做两件结构性处理：① 标识符原样（甲/乙/丙 + 标签里的 一/二/三，M3 的标识符铁律）；",
        "# ② 标识符型标签在长文本里统一替换成选项栏同一英文串（否则 state 与选项对不上，gold 不可达）。",
        "#",
        "# 纪律（与 v1 相同）：id 数字位与中文集一一对应；category/题型/选项数/gold 照抄不重生成；",
        "# choice 标签逐条验证英文码点序 == 中文槽位序（槽位不得漂移，漂了就剔题）；score 档位顺序照抄。",
        "#",
        f"# 本次：写出 {len(out_items)} 条｜**整条剔除 {len(dropped)} 条**｜事实 token 未存活 {len(fact_flags)} 条"
        f"｜标识符指代未统一 {len(ident_mentions)} 条",
    ]
    if dropped:
        header += ["#", "# 整条剔除明细（S3：不可用只剔除、不许人工改译文）："] + [f"#   {s}" for s in dropped]
    if fact_flags:
        header += ["#", "# 事实 token 未存活明细（须进 20 条抽检）："] + [f"#   {s}" for s in fact_flags]
    if ident_mentions:
        header += ["#", "# 标识符指代未统一明细（须进 20 条抽检）："] + [f"#   {s}" for s in ident_mentions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        for it in out_items:
            fh.write(json.dumps(it, ensure_ascii=False, separators=(",", ":")) + "\n")

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report = {
        "out": str(args.out),
        "sha256": digest,
        "n_items": len(out_items),
        "engine": {"key": args.engine, "model": args.model, "revision": rev, "weights": sha,
                   "num_beams": args.num_beams, "no_repeat_ngram_size": 3,
                   "max_length": "min(512, 3*len(sent)+40)，下限 64", "kind": "MT (non-LLM)",
                   "params": ENGINES[args.engine]["params"],
                   "probe": probe, "probe_ok": not NON_LATIN.search(probe),
                   "transformers": "4.44.2（隔离目录，经 PYTHONPATH 前置；venv-laya 的 5.17 未动）",
                   "download_source": "ModelScope 魔搭（modelscope download）",
                   "license": ENGINES[args.engine]["license"]},
        "n_out": len(out_items),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "fact_drift": fact_flags,
        "ident_mentions": ident_mentions,
        "term_audit": term_audit,
        "unusable_label_sets": {"/".join(k): v for k, v in UNUSABLE_LABEL_SETS.items()},
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {args.out}（{len(out_items)} 条）\nsha256: {digest}")
    print(f"写出 {len(out_items)} 条（整条剔除 {len(dropped)} 条）｜事实 token 未存活 {len(fact_flags)} 条")
    if args.print_manifest_entry:
        entry = {
            "path": str(args.out),
            "sha256": digest,
            "n_items": len(out_items),
            "language": "en",
            "status": "frozen",
            "note": (
                f"M4-1：`zh-evidence-v1.jsonl` 的**第二独立译源**对照集。M3 的 en-evidence-v1 由被评模型自己译"
                f"（译者＝考生），本集改用专用 MT 模型 {args.model}（非 LLM，权重 sha256 {sha[:16]}…，"
                f"经魔搭 ModelScope 下载，transformers 4.44.2 走隔离目录）重译同一批中文原件，中文侧一字未动 ⇒ "
                f"唯一被换的变量是译者。按 specs/M4.md §S3：**不做人工改译文**，不合用的题只**整条剔除**"
                f"（本 run 剔 {len(dropped)} 条，逐条列在文件头与报告的 dropped 里）；只做标识符原样与"
                f"「标识符型标签在长文本里统一替换」两件结构性处理。choice 标签逐条验证英文码点序 == 中文槽位序，"
                f"漂了就剔题。结构闸门输出见 scripts/check-en-parity.py。许可：{ENGINES[args.engine]['license']}"
            ),
        }
        print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
