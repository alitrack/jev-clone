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
| `positive` | 可选 | 二值化时哪个槽位是正类（只影响 `brier_binary`）。**没有缺省值**：不写就不参与 `brier_binary`（`n_binary` 会小于 `n_scored`），`noul` 由契约固定为 `"true"`。口径以 `metrics.rs` 为准（见本文件开头） |
| `provenance` | 可选 | 这道题为什么这么判——写给复核的人看，**必须能由 `state` 推出** |

四类（`category`，Q2 的能力拆分）：

- `evidence_judgment` — 材料支持不支持某个断言；
- `rule_application` — 把规则套到事实上；
- `candidate_selection` — 在若干候选里选最合适的；
- `missing_evidence` — 判断某关键材料是否缺失。

题集是**冻结**的：`eval/items/manifest.json` 记录文件 sha256 与条数，
`run --manifest` 在哈希不符时**拒绝计算**。改题集必须显式改 manifest（有意为之的动作），
不许悄悄改。

当前集合（`eval/items/manifest.json` 里逐条登记）：

| 集合 | 条数 | 状态 | 用途 |
|---|---|---|---|
| `zh-evidence-v1.jsonl` | **150** | `frozen` | **默认题集**，达到 `docs/design.md` M2 的规模目标 |
| `zh-evidence-v0.jsonl` | 45 | `starter` | M1 的起步集，**保留不动**：`eval/runs/` 里 M1 的报告要靠它的哈希才能复现 |

v1 的四类分布 38/38/38/36，题型 choice 98 / noul 32 / score 20（v0 是 35/8/2 —— score 只有 2 条且 gold 同为第 1 档，
等于那条通道此前没被测过）。id `0001–0045` 逐字承自 v0，仅 `source` 字段统一为 `authored-zh-v1`，所以 v0/v1 在这 45 条上可直接对照。

**偏差度量**（`scripts/validate-items.py` 可复现，v1 与 v0 对照）：

| 度量 | v0 | v1 | 含义 |
|---|---|---|---|
| choice gold 槽位 | A 13 / B 17 / C 5 | A 42(43%) / B 43(44%) / C 13(13%) | 恒答同一槽位的得分上限 |
| gold 是否为书写首位 | 23/35 = **66%** | 51/98 = 52% | 「恒取第一个选项」的得分上限 |
| noul 真假 | true 5 / false 3 | true 17 / false 15 | 少数类占比 38% → 47% |
| score gold 档位 | 1 档 **2/2** | 0 档 5 / 1 档 10 / 2 档 5 | 恒答第 1 档的得分上限 |

⚠️ v0 的 header 与 manifest 旧文曾称「gold 偏向 A 槽约 2:1」——**实测证伪**（实际 A 13 / B 17）。
v0 真正暴露的退化策略是「恒取书写第一个选项」（66%）。manifest 里的原说法已就地更正并注明。

两个脚本随仓走（`scripts/`）：

- `validate-items.py <items.jsonl>` —— 在 crate 之前把题集按它将被读入的方式校验
  （JSON 合法、id 唯一、gold 属于该题自己声明的槽位、`provenance` 存在），并打印上面那张偏差表与 sha256。**造题时每批必跑**。
- `merge-item-sets.py --source <名> --out <出> [--header <头>] <输入…>` —— 把若干题集并成一份冻结集并统一 `source`。
  纯函数式合并（按 id 排序、重复 id 直接失败），因为冻结点必须字节可复现：手改一行就是"悄悄不再可复现"的开始。

---

## 2. 三个子命令

```bash
# 服务端答案 → 预测行（predictions JSONL）
jev-eval import --items eval/items/zh-evidence-v1.jsonl \
                --answers eval/runs/<date>-answers.json \
                --out     eval/runs/<date>-predictions.jsonl

# 出报告（可加 --manifest 做哈希闸门，--out/--md 落盘）
jev-eval run --items eval/items/zh-evidence-v1.jsonl \
             --predictions eval/runs/<date>-predictions.jsonl \
             --manifest eval/items/manifest.json \
             --out eval/runs/<date>-report.json --md eval/runs/<date>-report.md

# 复现校验：从原始证据重算报告里每一个数字并逐项比对（阈值由 --tol 给出，默认 1e-12）
jev-eval verify --items eval/items/zh-evidence-v1.jsonl \
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
2. **报告里的每个数字都能重算。** `verify` 从题集与预测文件重算全部指标并与报告逐项比对。
   **审计阈值属于调用方**：`--tol`，默认 `1e-12`；报告里自报的 `verify_tolerance` 是被比较的**主张之一**，
   报得比我们松就记 mismatch。2026-09-19 之前 `verify` 反过来从被审计的报告里读阈值，于是「改掉 ECE +
   自报 `verify_tolerance: 1e9`」就能打印 `0 mismatches` 并 exit 0 —— 端到端脚本的第 6 步
   （`scripts/tamper-report.py`）就是这一步的否证测试，它必须以非零退出才算通过。报告不可复现 = 报告作废。

## 5. 已知未完成

- 题集已达 150 条（v1），但仍**全部为自撰题**：分数只能作同一端点上的相对对照，不是模型能力结论。
- 四类题的难度未做等值标定，"哪一层更难"仍属观察而非结论。
- 中文题集目前只覆盖证据类任务，未覆盖多跳/长材料。
- 退化策略的**基线**目前只有题集侧的度量（恒答同一槽位 / 恒取第一个选项，上限均约 50%）；把这两条
  真的当成策略跑一遍、给出它们的 ECE/Brier，仍未做。在跑出基线前，任何单次 ECE/Brier 数字都不足以说明模型好或坏。

## 6. `eval/runs/` 里的产物

每次端到端跑（`scripts/accept-m1-eval.sh`）落一套 `<UTC 时间戳>-{answers,predictions,report}.json[l,md]`，
**只追加不覆盖**，因为报告里的每个数字都要能追到产生它的那次原始结果。

| 运行 | 说明 |
|---|---|
| `20260919T141756Z` | **没有 report，这是正确的结果**：该次预测里有一条 `score` 概率和 = `1.000001`，`run` 按契约的 `1e-6` 容差拒绝计分。根因在服务端——它把每个概率**独立**四舍五入到 6 位小数，n 个槽位就有 ±n×5e-7 的漂移。已修（舍入后把残差补给答案 argmax 所指的那一格），并保留了这份"被拒"的原始记录 |
| `20260919T141944Z`、`20260919T142125Z` | 修复后的两次完整跑，均有 report；`verify` 逐项重算 0 不匹配 |
| `20260919T152343Z` | **v1（150 条）首次全量跑**，成绩见 §7；`verify` 572 项检查 0 不匹配，tamper 门拒收被改写报告 |

## 7. 最近一次全量成绩（v1 / 150 条 / 真端点 8014）

题集 `zh-evidence-v1.jsonl`（sha256 `2f0415d9…`），后端 `qwen3.8-27b`（SGLang TP2，10.10.10.115:8014），
逐题一次请求；命令见 `scripts/accept-m1-eval.sh`，原始产物 `eval/runs/20260919T152343Z-*`。

| 分层 | n | acc | bal_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|
| overall | 150 | 0.840000 | 0.880229 | 0.402757 | 0.236100 | 0.101340 | **0.043950** |
| candidate_selection | 38 | **1.000000** | 1.000000 | 0.076353 | 0.017161 | 0.009976 | 0.070311 |
| evidence_judgment | 38 | 0.815789 | 0.860000 | 0.411054 | 0.256571 | 0.128285 | 0.114427 |
| missing_evidence | 36 | 0.861111 | 0.796537 | 0.416784 | 0.231969 | 0.133493 | 0.127131 |
| rule_application | 38 | **0.684211** | 0.731548 | 0.707575 | 0.438483 | 0.167380 | **0.130299** |

错误归因（`scripts/error-digest.py`，它的准确率必须与报告一致 —— 0.8400，否则说明它复现槽位/argmax 规则复现错了）：

```
wrong 24/150        rule_application 12/38 (0.316)   evidence_judgment 7/38 (0.184)
                    missing_evidence 5/36 (0.139)    candidate_selection 0/38
by type             choice 15/98 (0.153)   score 6/20 (0.300)   noul 3/32 (0.094)
方向                否定被答成肯定 11 次；肯定被答成否定 3 次
noul 混淆            false->true 2 / true->false 1
score 档位偏移        -2:1  -1:2  +1:3
```

**必须一起念的三条附注**（只看 0.84 会读歪）：

1. **`candidate_selection` 是 38/38，方差为零 —— 这一层没有区分度。**它的 ECE（0.070）看着漂亮，
   但一个全部答对的层不可能区分模型好坏。这一层（多为「候选里哪个满足全部条件」的多选）对本端点太容易，
   要用它作证据须先加难度或换题型，**不能报成"该能力已达标"**。
2. **M3 的门「ECE ≤ 0.05」在未做任何标定时就已满足**（0.04395）。也就是说这个门按现在的写法
   在本题集上无法证明标定工作有效 —— 它是**已达成但无区分度**的门，引用 M3 前必须重述。
3. **本轮只有一个抽样**（每题一次请求，无重复采样、无置信区间）。24 个错误的方向如此集中（11:3），
   说明主导错误可能是**单一系统性偏差**（倾向于肯定断言）而不是随机散布 —— 这值得下一步专门验证，
   现在只是一轮的观察，不下结论。

其余口径不变：全为自撰题，分数只作同一端点上的相对对照，不是模型能力结论；四类难度未做等值标定。
