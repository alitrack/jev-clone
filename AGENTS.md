# AGENTS.md — jev-clone 工作约定

## 项目本质

本仓库复刻的是**接口契约**（state + 运行时 typed questions → 一次前向 → 概率），不是任何官方模型。
一切实现决定都以「不生成 token、答案被约束在选项集合内」为前提。

## 铁律

1. **先蓝本、后代码**：任何实现前先读 `docs/blueprints.md`；三份参考实现的本地副本在
   `/mnt/d/wsl2/tmp/jev-refs/`。不要凭记忆写契约字段。
2. **不复制无 license 的代码**：`daseinlabs/open-jev` 无 LICENSE，只可借鉴契约形状，不可逐行搬。
3. **槽位校验不可省**：答案槽位必须"单 token 且往返唯一"，且 `tokenize(prompt+A) == tokenize(prompt)+[slot]`。
   校验失败要**拒绝服务**，不是降级静默算错。
4. **不截断**：超长输入抛 422 类错误。官方预算 64k（state + 最长问题 ≤32k）。
5. **问题之间独立**：一个问题的答案不得进入另一个问题的 prompt（官方明文语义）。
6. **诚实措辞**：只说"输出被约束在给定选项集合内"，**禁止**写"不能幻觉""不会出错"。
   任何数字必须能回指原始结果文件。
7. **指标不合并**：按来源分层上报；accuracy / agreement / ECE / Brier 分开报；
   不把 TypeSafe 公开评测当 ground truth。
8. **纯 Rust 产品线**：`crates/` 不得引入 Python 运行时；训练脚本只允许放在 `train/`（一次性离线）。
9. **先查 crates.io 再手写**：手写是 fallback 不是首选。
10. **数字口径**：官方自报的 40–200× / 40–400× 只能在明确标注"厂商自报、自有 workflow 基准"时引用。

## 目录

```
crates/jev-core      契约类型 / 渲染器 / 槽位校验 / 概率与 confidence
crates/jev-backend   推理后端 trait（llama.cpp / candle / OpenAI 兼容）
crates/jev-server    axum HTTP /v1/systemone
crates/jev-eval      冻结题集 runner + 指标 + 复现校验
eval/sets/           题集（含 manifest 与内容 hash）
train/               M3 训练脚本（Python，离线，不进产品线）
docs/                design.md（设计）/ blueprints.md（蓝本研究）
```

## 提交前自检

单测全绿 → 契约示例 fixture 字段级对齐 → 评测数字能被复现脚本重算 → README 措辞检查（铁律 6）。
