# 开发环境怪癖（WSL，2026-09-19 实测）

本文件记录在这个开发机上**干活前必须知道**的三件事。每条都有实测证据，不是推测。
不写进 README（对使用者无关），只写在这里（对开发者/agent 有关）。

---

## 1. cargo 的 TLS 客户端是坏的 → 一律 `--offline`

**症状**：任何走网络的 cargo 命令失败，且**换镜像无效**：

```
[35] SSL connect error (TLS connect error: error:0A000126:SSL routines::unexpected eof while reading)
```
rsproxy.cn / mirrors.tuna / mirrors.ustc / index.crates.io **四种源形态全挂**，错误一模一样；
同一时刻 `curl` 打同一个 URL 连打 6 次全 200，`git ls-remote https://github.com/...` 也秒通。
强制 HTTP/1.1（`http.multiplexing=false`）、固定 TLS 1.2、伪装 UA —— 都不救。

**处置**：
- 所有 cargo 命令加 `--offline`
- **禁用 `cargo add`**（联网失败）；加依赖 = 手改 `Cargo.toml` + `bash scripts/fetch-deps-offline.sh`
- 该脚本从 `failed to download \`name vX.Y.Z\`` 抠出缺的包，用 curl 从 `static.crates.io` 补齐进
  `~/.cargo/registry/cache/<registry>/`，再重试离线构建。校验和仍由 Cargo.lock 把关
- `.cargo/config.toml` 只在**本仓**生效，全局配置未动

---

## 2. sigil 长跑会被 SGLang 以 `400 System message must be at the beginning.` 掐死 → 用 `tools/sysrole-proxy.py`

**根因（读 sigil 源码确认）**：sigil 会往对话中间插 `system` 角色消息——
`inject_goals_block()` 插在索引 1（`src/agent/mod.rs:381-396`）；上下文压缩摘要也是 system 角色、
落在消息列表中间（同文件 `:240`）。SGLang 严格校验 system 消息必须在**第 0 位**。

**后果**：长任务（实测约 16 次工具调用 / 30 分钟）被 400 掐死，**一个文件都没写就退出**。

**处置**：起代理，client 指向它：

```bash
python3 tools/sysrole-proxy.py          # 监听 127.0.0.1:8090 → 转发 115:8014
# sigil --base-url http://127.0.0.1:8090/v1
```
代理把非首位 system 改写成 user（加 `[system] ` 前缀，信息不丢），其余字节原样透传（含 SSE 流式分块）。
实测：同一 body 直连 8014 → 400；经代理 → 200。请求日志在 `/mnt/d/wsl2/tmp/jev-m0/proxy.log`，
每行含 `msgs=N` 与是否发生改写。

---

## 3. 115 两个端点的能力边界（选后端前必看）

| 端点 | 引擎 | 能读 logprobs？ | 能当 sigil 后端？ |
|---|---|---|---|
| `lan-gpu-host:8014` | SGLang（qwen3.8-27b，DFlash2 投机） | ✅ `/v1/completions` + `logprobs`；`/tokenize` 也可用 | ✅（经 §2 的代理） |
| `lan-gpu-host:8017` | NInfer（同模型，MTP，vision） | ❌ 回 `logprobs_not_supported` | ❌ 非平凡 prompt 下 sigil 空手退出 |

另外两条实测结论，直接决定读出头协议：

1. **必须走裸补全 `/v1/completions`，不能走 chat**：chat 模板先吐 thinking token，`max_tokens=1` 的那一个
   位置会被 `We` 之类占掉，答不上题；除非显式 `chat_template_kwargs={"enable_thinking":false}`。
2. **字母槽位不是审美，是正确性装置**：实测同一位置 `"yes"` / `"Yes"` / `" yes"` 是**三个不同 token**，
   用自然语言标签会让概率被拆散、排序出错。而 `POST /tokenize` 证明 `Answer:\n` = `[15666,25,198]`、
   追加 `A`/`B`/`Z` 各只加一个 token（32/33/57）且 prompt 段不重切。

---

## 4. 验收

```bash
scripts/accept-m0.sh              # 离线构建 + 单测 + 起服务打真 8014 + 20 项契约断言
scripts/accept-m0.sh --no-live    # 只跑构建与单测
```
