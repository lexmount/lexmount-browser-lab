# Qwen3-8B × LexBench-Browser

固定协议：

- 官方runner：`browseruse-agent-bench@ccd5fcbdfb975257b2ce38161dc9bc2ab294b420`
- Dataset：`LexBench-Browser / All / 210`
- Agent：`browser-use / qwen3_8B`
- Browser：`lexmount`或`local`
- 正式rollout并发：10
- Agent：40 steps、600秒、flash mode、structured JSON schema
- Judge：gpt-5.4、stepwise、每任务官方阈值
- 压力Stage 2 Judge并发：5

模型、Browser和Judge配置见[`config.yaml`](config.yaml)。运行入口统一为：

```bash
./scripts/run_lexbench.sh qwen3-8b \
  --env-file /data/wf/sxh/.env.lexbench \
  --runtime-root /data/wf/sxh \
  --backend all \
  --mode all \
  --stage all
```

正式结果见[`docs/eval_reports/lexbench/`](../../docs/eval_reports/lexbench/README.md)。

## 受控浏览器对照

[`config.controlled-egress.yaml`](config.controlled-egress.yaml)将Local Chrome和Lexmount
浏览器分别指向`LEXBENCH_LOCAL_PROXY_SERVER`和Lexmount external proxy。配套的
[`task_sets/controlled-parity-20.txt`](task_sets/controlled-parity-20.txt)为固定的、非登录、
region/type分层20题子集。运行时使用
[`scripts/run_qwen3_8b_lexbench_controlled_pair.sh`](../../scripts/run_qwen3_8b_lexbench_controlled_pair.sh)
完成两端串行rollout、官方gpt-5.4 stepwise Judge、资源采样和配对审计。代理地址和凭据只通过
环境变量传入，不能写入配置或提交到仓库。
