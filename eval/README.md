# `jev-eval` — 校准评测的口径与用法

本目录是 M1-2 的落点：把「答案对不对」升级成「答案对不对 **以及它自己知不知道自己对不对**」。

**口径的最终权威是 `crates/jev-eval/src/metrics.rs` 的模块文档**；本文件是它的散文复述。
两者不一致时，以代码为准，本文件算 bug（`metrics.rs` 第 3 行就是这么约定的）。

---

## 1. 冻结题集 `eval/items/*.jsonl`

一行一个 JSON 对象；`#` 开头与空行忽略。字段：

| 字段 | 必填 | 含义 |
|---|---|---|
| `id` | ✓ | 题号，评测与预测文件靠它对账 |
| `category` | ✓ | 分层键之一，见下 |
| `source` | ✓ | 题集来源标识（也是分层键，**永不合并**） |
| `state` | ✓ | 模型看得到的材料（中文，自足） |
| `question` | ✓ | `jev-core` 契约里的 `Question`（`choice` / `score` / `noul`），JSON 原样 |
| `gold` | ✓ | 标准答案：`choice` 为选项标签；`score` 为档位索引（整数）；`noul` 为 `true`/`false` |
| `positive` | 可选 | 二值化时哪个槽位是正类（只影响 `brier_binary`）。缺省：`choice` 为第一个槽位，`noul` 为 `"true"` |
| `provenance` | 可选 | 这道题为什么这么判——写给复核的人看，**必须能由 `state` 推出** |

四类（`category`，Q2 的能力拆分）：

- `evidence_judgment` — 材料支持不支持某个断言；
- `rule_application` — 把规则套到事实上；
- `candidate_selection` — 在若干候选里选最合适的；
- `missing_evidence` — 判断某关键材料是否缺失。

题集是**冻结**的：`eval/items/manifest.json` 记录文件 sha256 与条数，
`run --manifest` 在哈希不符时**拒绝计算**。改题集必须显式改 manifest（有意为之的动作），
不许悄悄改。

当前集合：`zh-evidence-v0.jsonl`，45 条，`status: starter` —— 设计目标（`docs/design.md` M2）
是 ≥150 条中文，**本集未达标**，manifest 里如实记为 starter 与 `target_n_items: 150`。
已知偏差写在题集头部注释与 manifest 的 `note` 里（gold 在 A/B 槽位、true/false 上不均衡），
扩集时必须修。

---

## 2. 三个子命令

```bash
# 服务端答案 → 预测行（predictions JSONL）
jev-eval import --items eval/items/zh-evidence-v0.jsonl \
                --answers eval/runs/<date>-answers.json \
                --out     eval/runs/<date>-predictions.jsonl

# 出报告（可加 --manifest 做哈希闸门，--out/--md 落盘）
jev-eval run --items eval/items/zh-evidence-v0.jsonl \
             --predictions eval/runs/<date>-predictions.jsonl \
             --manifest eval/items/manifest.json \
             --out eval/runs/<date>-report.json --md eval/runs/<date>-report.md

# 复现校验：从原始证据重算报告里每一个数字并逐项比对
jev-eval verify --items eval/items/zh-evidence-v0.jsonl \
                --predictions eval/runs/<date>-predictions.jsonl \
                --report eval/runs/<date>-report.json
```

`--answers` 的形状就是 `SystemOneResponse`：`{"model"?, "answers": {"<id>": Answer}, "usage"?}`。
`import` 会把**没有答案**的题写成 no-answer 行并以警告列出（覆盖度不许被藏起来），
并额外报告「服务端自己的 decision 与它自己分布的 argmax 不一致」的题目——这是免费的契约自检。

---

## 3. 指标口径（冻结）

- **predicted slot** = 概率向量的 `argmax`，平局取题目声明的槽位顺序里更靠前的那个；
  没有概率向量的行取预测文件里的 `label`。
- **准确率** = 预测正确的行数 / 有预测的行数。
- **平衡准确率** = 各**实际出现过的 gold 类**的召回率的平均；没有 gold 的类被排除
  （这是 SemIf `evaluate.py` 的约定，即 macro-recall）。类别多于两类时含义不变。
- **NLL** = 有评分行上 `-ln(max(p_gold, 1e-12))` 的平均。
- **Brier 两种都给**（选项多于两个时「Brier 分数」是有歧义的）：
  - `brier_multiclass` = `Σ_k (p_k − 1[gold == slot_k])²` 的平均；
  - `brier_binary` = 仅在有 `positive` 的行上 `(p_positive − 1[gold == positive])²` 的平均，
    `n_binary` 记录参与行数；
  - 两槽位时二者精确相关：`multiclass = 2 × binary`（有测试断言这一点）。
- **confidence** = `max(p)`，即预测槽位背后的概率质量。
- **ECE** = `Σ_b (n_b / N) · |acc_b − conf_b|`，`bins` 个等宽箱；
  箱号 `min(bins−1, floor(conf · bins))`，所以 `conf == 1.0` 落在最后一箱而不溢出。
  `N` 为有评分行数。
- **可靠性曲线** = ECE 所累加的每箱 `(lower, upper, n, mean_confidence, accuracy)`；空箱省略。

没有概率向量的行计入 `n_items` / `n_missing` 与准确率，但**不参与** NLL / Brier / ECE；
`n_scored` 明确给出真正参与的行数。

## 4. 必读的两条诚实性约定

1. **覆盖度不许被藏。** 缺答案的行、没有概率的行都在报告里单列计数（`n_missing` / `n_scored` /
   `n_binary`），且 `import` 会把缺答案的题逐条列名。
2. **报告里的每个数字都能重算。** `verify` 从题集与预测文件重算全部指标并与报告逐项比对
   （容差默认取报告自己记录的值，`--tol` 可覆盖）。报告不可复现 = 报告作废。

## 5. 已知未完成

- 题集只有 45 条（目标 150），且 gold 分布有偏差 —— 见上。
- 中文题集目前只覆盖证据类任务，未覆盖多跳/长材料。
- 校准指标的**参考基线**（例如「模型恒答最可能槽位」的退化策略得分）尚未跑；
  在跑出基线前，任何单次 ECE/Brier 数字都不足以说明模型好或坏。
