#!/usr/bin/env python3
"""中英对照集的**机械化质检谓词**（单一实现，翻译脚本与冻结尾门共用）。

为什么要有这个模块：M4-1 的第一版把「数字核对」写在 `translate-mt.py` 里，结果
(a) 只活在翻译脚本内部，**冻结尾门不做事实核对**——等于英文集落盘后没人再验；
(b) 判据太粗：`norm_for_facts` 把中文「一**个**」也归一成 `1`，`1400` 与 `1,400`
    对不上，`180 万` 与 `1.8 million` 对不上。实测 150 条里报了 32 条「丢 token」，
    逐条归因后**大部分是写法差异**（19 个 token 是千分位、若干是单位改写），
    真正丢内容的只有阶梯水价那组（`181 至 260` → "161 to 260"、"260 以上" → "above 250"）。

因此判据改为**数值规范化比对**：两边先把数字抽成「带量纲的数」，再比集合。
`万/亿` 与 `thousand/million/billion` 都折算成同一个十进制值，千分位先去掉。
这样 `1400`≡`1,400`、`180 万`≡`1.8 million`，而 `181`≢`161` 仍会被抓出来。

范围（写在明处）：本模块**只**判「译文有没有把事实搞错」（数字/编号），
**不**判语义正确性——`单项不低于 100 万` 被译成 "no fewer than 1 million individuals"
属于语义走样，机器判不了，归 S3 的人工抽检（并如实记在报告里）。
"""
import re

CJK_RE = re.compile(r"[\u3400-\u9fff]")
ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{1,}")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

MONTHS = {m: str(i + 1) for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
EN_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
}
# 量纲词表：只收「不会跟单位混淆」的。故意不收 `k`/`m`（可能是米/千米）。
#
# `wan` 是**本项目英文集自己的写法**：M3 的手工对照集 `en-evidence-v1.jsonl` 把「万元」写成
# "500 wan yuan"（不转成 million），而 M4 的 MT 集写成 "5 million"/"5,000,000"。两边都得认，
# 否则同一份中文会被判成两种事实（实测：不认 `wan` 时 M3 冻结对子会误报 14 条事实漂移）。
SCALES = {"亿": 10 ** 8, "千万": 10 ** 7, "百万": 10 ** 6, "万": 10 ** 4, "wan": 10 ** 4,
          "billion": 10 ** 9, "bn": 10 ** 9, "million": 10 ** 6, "thousand": 10 ** 3}
SCALED_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)*)\s*(亿|千万|百万|万|wan|billion|bn|million|thousand)?",
    re.IGNORECASE)
# 「N 折」是中文折扣写法（9 折 = 原价 ×0.9），英文常写成 "10% off" —— 这是**写法约定**，
# 不是数值事实，故数值核对里先摘掉；折扣有没有被译出来，由 `discount_lost` 单列一条软判据。
ZHE_RE = re.compile(r"\d+(?:\.\d+)?\s*折")
DISCOUNT_EN = re.compile(r"discount|off\b|percent|%", re.IGNORECASE)
# 时刻写法：中文 `14:20`／`18:40`，MT 常写成 "2:20 p.m."／"3pm"（12 小时制）。这是**写法约定**，
# 不是数值事实（同一时刻），故数值核对前先摘掉；也避免 `15:00`→"3pm" 被当成丢了 15。
CLOCK_RE = re.compile(
    r"\d{1,2}\s*[:：]\s*\d{2}(?:\s*[ap]\.?\s?m\.?)?|\d{1,2}\s*[ap]\.?\s?m\.?", re.IGNORECASE)


def norm_en_words(text: str) -> str:
    """英文侧：月份与数词归一成阿拉伯数字（`June`→6、`four`→4）。

    只做 en→数字这一个方向：中文侧不做数词归一——实测「一**个**可离线使用」会被
    误归一成 `1`，英文写成 "an offline…" 就被判成丢了 token（0024 的假警）。
    """
    for word, num in {**MONTHS, **EN_NUM_WORDS}.items():
        text = re.sub(rf"\b{word}\b", num, text, flags=re.IGNORECASE)
    return text


def num_values(text: str) -> set[float]:
    """抽出文本里**带量纲的数值**（千分位/万/亿/million 都折算到同一量纲）。"""
    out = set()
    cleaned = CLOCK_RE.sub(" ", ZHE_RE.sub(" ", text))
    for raw, scale in SCALED_RE.findall(norm_en_words(cleaned)):
        v = _to_float(raw)
        if v is None:
            continue
        if scale:
            v *= SCALES[scale.lower()]
        out.add(round(v, 6))
    return out


def _to_float(raw: str) -> float | None:
    """解析数字串：逗号既可能是千分位（`1,400`）也可能是小数（`0,3`，欧式写法）。

    判据：逗号后正好 3 位且前面不是空的 → 千分位；否则当小数点。
    实测 MADLAD 会把中文 `0.3%` 写成 "0,3%"，按千分位解析会得到 3（差 10 倍）——这条是为此写的。
    """
    raw = raw.strip()
    try:
        if "," in raw and "." in raw:
            return float(raw.replace(",", ""))
        if "," in raw:
            head, _, tail = raw.rpartition(",")
            if len(tail) == 3 and head and head.replace(",", "").isdigit():
                return float(raw.replace(",", ""))
            return float(raw.replace(",", "."))
        return float(raw)
    except ValueError:
        return None


def fact_tokens(text: str) -> set[str]:
    """编号型 token（`L2`、`P95`、`QX1Q`）——它们被占位符保护，必须原样存活。"""
    return {t for t in ID_RE.findall(text) if len(t) >= 2 and not CJK_RE.search(t)}


def facts_lost(zh_text: str, en_text: str) -> list[str]:
    """中文侧的事实（数值 / 编号 token）是否都在英文里存活。

    数值按**量纲比对**（`1400`≡`1,400`、`180 万`≡`1.8 million`）；编号 token 按
    「不区分大小写的子串」判存活。返回**丢失清单**（空 = 全存活）。
    """
    lost = sorted(f"{v:g}" for v in num_values(zh_text) - num_values(en_text))
    en_low = en_text.lower()
    for t in fact_tokens(zh_text):
        if t.lower() not in en_low:
            lost.append(t)
    return lost


def discount_lost(zh_text: str, en_text: str) -> bool:
    """中文写了「N 折」但英文里连折扣字样都没有 —— 软判据（进 notes，不致命）。"""
    return bool(ZHE_RE.search(zh_text)) and not DISCOUNT_EN.search(en_text)


REPEAT_RE = re.compile(r"\b(\w+)\b(?:[,\s]*\1\b){4,}", re.IGNORECASE)


def degenerate_reason(zh_text: str, en_text: str) -> str | None:
    """机械退化判据：复读机 / 长度爆炸（判「译文坏了」，不是「译文不好」）。

    MADLAD-3B 在 2 字标签上会陷入复读机：实测 `否/是` → "No, no, no, …"×140。
    """
    if REPEAT_RE.search(en_text):
        return "同一词连续重复 ≥5 次"
    if len(en_text) > max(400, 8 * len(zh_text)):
        return f"长度爆炸（中文 {len(zh_text)} 字符 → 英文 {len(en_text)} 字符）"
    return None


if __name__ == "__main__":  # 自测：把 M4-1 实测的几种情形钉住
    cases = [
        ("阶梯水价 181 至 260 立方米", "161 to 260 cubic meters", ["181"]),
        ("合同金额 180 万", "contract value of 1.8 million", []),
        ("金额 1400 元", "the amount of 1,400 yuan", []),
        ("用水 300 立方米", "300 cubic meters of water", []),
        ("一个可离线使用的工具", "an offline tool", []),
        ("编号 L2 与 P95 延迟", "L2 and P95 delay", []),
        ("编号 L2", "the second tier", ["L2"]),
        ("四年", "four years", []),
        ("6 月", "June", []),
        # 本项目英文集的两种「万」写法都必须认（M3 手工集用 wan，M4 的 MT 集用 million/分位）
        ("注册资本 500 万元", "registered capital of 500 wan yuan", []),
        ("注册资本 1200 万元", "registered capital of 12 million yuan", []),
        # 「打 9 折」↔ "10% off" 是写法约定，数值核对应放行
        ("平台券再打 9 折", "a further 10% discount", []),
        # 欧式小数逗号（MT 实测把 0.3% 写成 0,3%）与 12 小时制时刻都要放行
        ("崩溃率 0.3%", "a crash rate of 0,3%", []),
        ("6 月 6 日 14:20 揽收", "picked up at 2:20 p.m. on 6 June", []),
        ("4 月 10 日 15:00 表态", "spoke at 3pm on April 10", []),
    ]
    bad = 0
    for zh, en, want in cases:
        got = facts_lost(zh, en)
        ok = got == want
        bad += not ok
        print(f"{'✓' if ok else '✗'} {zh!r} vs {en!r} → {got}（期望 {want}）")
    print(f"\n自测：{'全部通过' if not bad else f'{bad} 条不符'}")
