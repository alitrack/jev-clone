# 蓝本研究：三份开源复现 + 官方契约（源码级）

> 2026-09-19 ｜ 全部结论来自**克隆到本地逐行读过的源码**，不是 README 转述。
> 本地副本：`/mnt/d/wsl2/tmp/jev-refs/{jevlike,SemIf,open-jev}`（`--depth 1`）
> 官方文档 markdown 快照：`/tmp/jev/official/md/*.md`

---

## 1. `vinnylarouge/jevlike`（897★，MIT，生态里最早）

**它是什么**：一次前向的"选项打分器"（不是生成器）。

**源码事实**（`jevlike/jevlike/model.py`）

- `AttentionHead`：`query = W_q(options)`，`key/value = W_kv(context)`，`scores = einsum(q,k)/√rank` → mask 掉 padding → softmax → 加权 value → `logits = (query * attended).sum(-1) / √rank`，最后把非选项位置 mask 成 `-inf`。
  → **槽位是"位置"，不是"token"**：选项向量来自全局池化，天然支持多 token 选项。
- `FrozenTransformerScorer`：`AutoModel` 编码器 `requires_grad_(False).eval()`，**唯一被训练的是 head**；encoder 走 `torch.no_grad()`。选项侧逐选项池化后 reshape 回 `[batch, n_options, hidden]`。
- `TinyScorer`：`Embedding(257)` 字节级编码 + `Embedding(context_tokens)` 位置 + 同一个 head。
- `load_checkpoint`：只存 head 参数 + encoder 名字（`config`），加载时校验 missing/unexpected。
- 训练：`train.py`（95 行），监督式；`--encoder hf --hf-model ...` 可换成冻结的 HF 模型。

**它证明了什么**：冻结编码器 + 小 head 的"线性探针"路线**便宜且稳**（特征可预算缓存，头部训练秒级）；**上限由编码器表征是否已含判别信号决定**。

**它没做什么**：没有校准（score 没有"概率"语义）；没有运行时 criteria 的 typed question 契约；作者明说 "not a copy of Jev" / "did not show equal quality with Jev"。

---

## 2. `TheoLeeCJ/SemIf`（原 openjev，1608★，MIT，唯一对齐官方公开评测）

**它是什么**：两套对照系统 + 一整套**冻结评测协议**。质量与诚实度最高的一份蓝本。

### 2.1 直读路线（`src/semif_phase1/direct.py`，79 行）

```text
prompt = chat_template(messages, add_generation_prompt=True, enable_thinking=False)
ids    = encode(prompt)                      # 超长直接抛错，不截断
slots  = _slot_ids(tokenizer, n_options)     # 'A'..'Z' 必须是「单 token 且往返唯一」
断言   : encode(prompt + letter) == ids + [slot_token]   # 答案边界不得改变 tokenization
前向   : model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
读取   : vocabulary[slots] → softmax → probabilities
落盘   : prompt_sha256 / prompt_version / input_tokens / forward_seconds /
         readout="restricted to declared answer slots" /
         probability_status="conditional option score; uncalibrated as decision confidence"
```

**这三条断言是整份蓝本里最值钱的工程细节**——它把"模型会不会答非所问"变成了**可静态校验的契约**：槽位不是单 token、或加答案后 tokenization 变了，就**拒绝服务**而不是默默算错。

### 2.2 重排路线（`reranker.py`）

用 Qwen3-Reranker-4B 的 yes/no 原生契约：每个候选写成一个 query/document 相关性命题，取 `logit(yes) - logit(no)`，再在选项间做 softmax。SemIf 自己声明：**最后这步归一化是他们的比较规则，不是上游的校准契约**。

### 2.3 评测协议（`benchmarks/manifests/metric-contract.json` + `docs/METHOD.md`）

- **706 行冻结矩阵**：自建 144 / WANLI 256 / TypeSafe 公开子集 102 / Every 公开 artifact 204。
- 事前冻结：prompt、ID、标签、任务语义、revision、指标**全部在跑重排输出之前冻结**。
- **禁止合并成一个总分**：按源分层上报；硬标签用全分母准确率 / 平衡准确率 / macro F1 / NLL/Brier + 源分组 bootstrap 区间；检索单独用 Recall@1/3、MRR；TypeSafe 行用 equal-case macro（避免"问题多的 case 主导"）。
- **诚实边界**：TypeSafe 对照是"能本地对齐的 102 行"（官方广告是 711 行聚合，且**没有跑真 Jev**）；Jev/Opus/Sol 的数字是**读公开发布记录**得来的。
- 复现校验：`verify_published.py` 让论文/README 里每一个数字都能被 `results/raw/*` 重算出来（`close()` 容差 5e-10）。

### 2.4 实测数字（我们 M1/M2 的对照基线）

| 项 | 数字 |
|---|---|
| 模型阶梯（native BF16 直读） | Qwen3-0.6B 0.440 / MiniCPM5-2B 0.686 / **Qwen3.5-4B 0.813**（自建 144 行平衡准确率） |
| TypeSafe 公开 102 行 equal-case agreement | **Qwen3.5-4B 0.845 vs 公开 Jev 0.883**（差 3.8pp）；TV 距离 0.177 vs 0.127 |
| 扰动 | 选项反转 0.813（10 次翻转）/ 措辞包裹 0.706（9）/ 无关上下文 0.821（4） |
| 缺失证据 36 行 | 两个系统各出现 1 次"分数 >0.8 的非 insufficient" ⇒ **分数不能当操作级校准** |
| 速度（1×3090，777 判断） | fresh batch1 333.1 s（**2.33 判断/s**，state p50 8.99 s）→ 串行前缀 72.3 s → **并行 suffix 38.8 s（20.03 判断/s，p50 1.05 s）** |
| 决策 vs 生成（21 准则） | 直读 **1.023 s** vs 最强朴素基线（只要求有序 JSON 数组）**5.332 s**（111 token，首 token 0.489 s），两法 argmax 一致 18/21 |
| 更严格基线 | 要求 minified 无空白数组 → 3/3 跑爆 128 token 上限，**记为失败而非用来放大倍数** |

### 2.5 它自己列的"没做到"

架构/并行采样器、**RLCD 训练**、**可用于生产阈值的校准概率**、frontier 能力、官方延迟成本。→ **这正是本项目的三个主攻方向**。

---

## 3. `daseinlabs/open-jev`（32★，无 license⚠️）

**它是什么**：MLX + Gemma-3-4B 的 one-pass scorer，**实现了 `POST /v1/systemone`**。契约实现写得最完整（pydantic 逐字段），是 M0 的直接参照。

**源码事实**（`openjev/systemone.py`，190 行）

- 请求：`ChoiceQuestion.criteria` 是 `dict`（2..255），`ScoreQuestion.criteria` 是 `list`（≥2），`NoulQuestion.criteria` 可选且 key 必须 ⊆ {true,false}（validator 强制）。
- 渲染：`State:\n…\n\nQuestion:\n…\n\n` + `Choose exactly one option.\nOptions:\n- key: desc` + `\n\nAnswer:\n`（Score 用 `0: …` 编号 + "answer with the level number only"；Noul 可带 `Answer yes when: / Answer no when:`）。
- 应答：`ChoiceAnswer(choice, probabilities, confidence)`、`ScoreAnswer(score=Σ i·pᵢ, confidence, legend, probabilities)`、`NoulAnswer(noul=p_yes)`、`Usage(input_tokens, output_tokens)`。
- **`confidence = 1 - H(p)/log(n)`**，源码注释**自认**这是对 TypeSafe "distribution spread" 定义的**近似**。
- 明说：`Nothing is generated.`；每个问题一次前缀共享的批量前向。

**⚠️ 两个已知问题**：仓库**没有 LICENSE**（不要复制其代码，只借鉴契约形状）；README 里的免责声明在 2026-09-17 之后的重写中被删掉了（`ecosystem/FACTS.md` 有逐字记录）。

---

## 4. 官方契约（`docs.typesafe.ai` 快照，逐字段）

| 面 | 事实 |
|---|---|
| 端点 | `POST https://api.typesafe.ai/v1/systemone`，`Authorization: Bearer` |
| 请求 | `state`(string\|object\|array) + `model` + `questions`(map<id, Question>)；**id 不送给模型、不参与推理** |
| 三原语 | `choice`（criteria 为 map，≥2，≤255）/ `score`（有序数组，≥2）/ `noul`（可选 true/false 描述） |
| 应答 | `choice`→`choice`+`probabilities`+`confidence`；`score`→`score`+`legend`+`probabilities`+`confidence`（**score 可落档位之间**）；`noul`→`noul`（**无 confidence**） |
| usage | `{input_tokens, output_tokens}` |
| 错误 | 401 / 422 / 429(Backoff) / **529 Overloaded** |
| 预算 | 单请求 **64k**，其中 state + 最长问题 ≤ **32k**（约 15 万英文字符）；**不截断** |
| 并行性 | 同一请求内**所有问题并行**，"adding questions barely changes the response time"；问题之间**答案独立**，不得互为上下文 |
| 官方建议 | 一次问全（推测式扇出，"问一个可能用不上的问题几乎是免费的"）；复杂判断拆成多个问题再在代码里加权（composite scoring） |
| confidence | 只给 0..1 一个数 + 完整 probabilities；官方明说 **"you are never locked into our definition"**（即公式可替换） |
| 定位原话 | "Ask for a judgment a knowledgeable person makes in a second given the right context"；"Analyze this message and determine the best course of action" **不是** System One 的活 |

**官方自己承认的失败模式（jaggedness 页，9 条，我们的 README 必须引用）**：字面理解 / 不是计算器且数东西不可靠 / 日期当文本读 / 多跳掉分 / **大 state = context rot** / **对抗性内容能移动答案** / 矛盾指令会乱 / **结构不变量不成立（P(退款)=0.72 与 P(不退款)=0.47 同时出现）** / 不会生成。

---

## 5. 结论：我们做什么、不做什么

| | 三份蓝本已做 | 本项目要做 |
|---|---|---|
| 契约形状 | ✅ 全部复刻 | 字段级兼容 + 官方 4 个示例当 fixture |
| 一次前向读概率 | ✅ | 同（槽位校验器跨 tokenizer，含中文） |
| 前缀复用 | ✅ 实测过（8.6×） | 写进架构硬指标（≥3×） |
| 评测 | ✅ SemIf 的 706 行冻结矩阵 | 复刻该协议 + **中文题集** |
| **校准** | ❌ 全部留白（SemIf 自己列为 not reproduced） | **第一指标（ECE/Brier/可靠性曲线）** |
| **训练** | jevlike: 冻结+head；SemIf: 无 | Route A → Route C(LoRA) + **温度标定** |
| 中文 | ❌ 未评测 | 中文优先，与英文分开上报 |
