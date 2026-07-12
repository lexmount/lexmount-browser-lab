# Preflight 2026-07-12

## Ready

- Benchmark 固定为 `browseruse-agent-bench@b9d3dc655aec3cd10fd1fb86cfb7678bb9a27399`。
- LexBench-Browser 数据集 210 条，SHA-256 为
  `b2e8626b1554decce4f0ca6b4aa463f24ec9667836c74b797666d34e2f6b90fe`。
- OpenAI-compatible `/models` 返回 `gpt-5.5`。
- 两套 Lexmount project 的鉴权和 session 创建均可用。
- 英文 project c20：20/20 创建成功，P95 创建时间 35.485 秒，清理后 active=0。
- 中文 project c20：20/20 创建成功，P95 创建时间 43.362 秒，清理后 active=0。
- GPT-5.5 真实 Lexmount smoke（任务 6）：
  - browser-use 0.13.4，中文 profile 路由正确
  - 6 steps，105.242 秒，42,115 tokens，agent cost `$0.23505`
  - gpt-5.4 stepwise Judge：61/100，框架阈值判定通过
  - 轨迹、API logs、Judge summary 均成功落盘，清理后 active=0

## Capacity finding

英文 project 的 c40 探针不是旧报告中的“超过 20 立即 quota 失败”：

- 同时观察到 27 个 active session，证明账号容量已高于 20。
- 其余 13 个创建请求在 180 秒以上仍未完成，也未快速返回 quota 错误。
- 中止后已核对并清理全部本轮 session，active=0。

因此容量判定必须同时报告“成功创建数、创建延迟、排队超时和残留清理”，不能只看
HTTP create 是否接受。端到端正式并发先以 c20 为已验证稳态点，c40 需要在有界
poll timeout 下重测。

## Not ready

- 5090 主机 `wf@ubuntu` 当前 SSH 22 端口超时；NetBird 能解析 peer，但没有有效
  WireGuard handshake。Local Chrome、systemd cgroup 和资源采样尚无法在目标机验证。
- 目标机代理/国际站出口仍需确认；本地 Chrome 不会自动继承系统代理。
- `config_snapshot.json` 会脱敏 API key，但仍保留 project ID。原始快照不得提交本仓库；
  对外共享前继续做 artifact sanitize。
