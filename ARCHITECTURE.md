# 目标：
- 评估qwen3-8B模型在多个web agent benchmark上的分数
- 评估出 lexmount browser这个云浏览器后端相比WebArena-Lite默认Playwright Chromium的差异
- 评估出 lexmount browser这个云浏览器后端相比本地Chrome的差异
- 针对每一个benchmark，产出可以一键运行benchmark评测的脚本。一键运行，评测，跑完后输出每个模型的分数、异常任务清单和可审计报告，以及资源效率指标报告。脚本支持传入参数 --backend xx 用于指定浏览器后端。
- 产出压力测试下的资源效率指标报告（1. CPU 效率，浏览器 rollout 时 cpu 的占用
2. GPU IDLE 的降低
3. 内存占用
4. 最大并发量 + 吞吐 ）。目标是验证 云浏览器 lexmount browser 相比本地浏览器 local Chrome/Playwright Chromium 的资源效率提升。

## 产出内容
- 一键运行的评估脚本（针对每一个benchmark，写一个可以一键运行评测这个benchmark的脚本）
- 详细的评估报告（分数 + 资源效率指标）

# Benchmark
1. webarena-lite (https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite)
2. LexBench-Browser (https://huggingface.co/datasets/Lexmount/LexBench-Browser)
3. Online-Mind2Web (https://github.com/OSU-NLP-Group/Online-Mind2Web)

# 评估模型
- qwen3-8B模型
模型API：
```
OPENAI_API_KEY=sk-abc123
OPENAI_BASE_URL=http://10.2.131.41:18088/v1
OPENAI_MODEL=qwen3_8B
```

# 评估流程
## 评估webarena-lite

1. 基于官方提供的docker环境（WebArena-Lite默认Playwright Chromium），评估 qwen3—8B模型；
2. 采用 lexmount browser作为浏览器后端，评估 qwen3—8B模型；

经过以上的评估流程后，就可以看出在同一个task输入和verfier的条件下，两个浏览器后端，qwen3-8B模型效果上的变化。可以评估出 lexmount browser这个云浏览器后端相比WebArena-Lite默认Playwright Chromium的差异。以及资源效率是否有提升！
lexmount 浏览器配置信息(API-KEY, PROJECT-ID)见：@env.md

其余内容参考 @docs/benckmarks/webarena_lite.md


## 评估 LexBench-Browser

按照官方脚本LexBench 的评估逻辑是：LLM judge 看截图/轨迹/最终回答，然后按 100 分制逐项打分；最后用每条任务自己的 score_threshold 判 pass/fail。
官方是否存在可以 一键运行的评估脚本？仔细检查
必须以官方为准！

我们需要评估出
lexmount browser 对比本地 local Chrome，两种环境的差异

其余内容参考：
@docs/benckmarks/lexbench.md

要求：
- 首先用 Lexmount browser 作为浏览器backend，评估出 qwen3_8B的分数
- 把 browser backend 换成 local，评估出 qwen3_8B的分数
lexmount 浏览器配置信息(API-KEY, PROJECT-ID)见：@env.md


## 评估 online-mind2web
同样的，本地local 和 lexmount云浏览器的对比
详细内容：
@docs/benckmarks/online_mind2web.md

# 注意事项
- 必须以官方的测试流程、测试指标为准！我们的脚本更多的只是为了增加 资源埋点的逻辑，监控资源效率！
- 5090 不再自建 WebArena 网站，只做 runner；另起一台官方 AMI EC2 专门跑 WebArena websites；5090 通过 WEBARENA_SERVER=<EC2公网IP或域名> 连接过去跑 WebArena-Lite。
- 你现在是在一台mac笔记本上，你必须在5090服务器上进行评测，文档中说的本地 local指的是 5090服务器这台机器：
```
ssh wf@10.2.131.41
password: waple0820
工作目录：/data/wf/sxh
```

# 你应该做的事
你需要做的是：
- 正式评测；并发10 个tasks （这是在评测速度与可用资源之间的折中值）。
- 压力测试；等到正式评测结束后，压力测试 20个task和50个task的并发（从所有tasks中，选择前20个和前50个task）
- 压力测试需要统计的指标：每一种测试下（20，50），在云浏览器和本地机器两种浏览器后端环境下，监控5090这台服务器的资源指标（1. CPU 效率，浏览器 rollout 时 cpu 的占用
2. GPU IDLE 的降低
3. 内存占用
4. 还有吞吐））

正式评测需要统计的指标：
- success%
- avg steps
- avg e2e(s)


一旦正式评测开启，你需要创建一个轮询/巡检机制，每隔5分钟检查评测的进度和状态，需要：
- 当前任务给出一条简短状态报告（即使无变化）
- 官方 runner 进程、官方 Stage/marker、sidecar CSV 连续性；
- 说明正推进、完成结果（如已有），以及浏览器导航/交互、超时、循环或重试信号；
- Qwen API 与 JSON-schema 输出路径可达；
- Judge API 可达；
- Lexmount browser 路径是否有连接、会话或导航错误。每条判断需区分“已验证”“尚不能证明”。
- 若发生异常、进程退出，立即优先报告；（比如：qwen API不可达、judge LLM API不可达，lexmount browser不通等问题）


# Intro
```
Playwright Chromium
  浏览器跑在 5090 / training node 上
  浏览器 CPU/RAM 占用本机资源

Lexmount cloud SDK
  浏览器跑在 Lexmount 云端
  5090 上主要只有 SDK client / CDP client
```


# 相关参考资源
- https://github.com/web-arena-x/webarena/tree/main/environment_docker
- https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite#setup-webarena-lite-environments
- https://github.com/THUDM/VisualAgentBench/tree/main/VAB-WebArena-Lite
- https://huggingface.co/datasets/Lexmount/LexBench-Browser
- https://github.com/lexmount/browseruse-agent-bench
- https://docs.bubench.lexmount.io/zh/benchmarks/online-mind2web
