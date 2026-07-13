工作目录：
/data/wf/sxh/workspace/LexBrowserEnv

训练框架：
https://github.com/nvidia-nemo/rl
https://github.com/NVIDIA-NeMo/Gym


已跑通的配置：
/data/wf/RL/nemo_rl

这个路径，跑通了：nemo-rl+nemo-gym+lexmount browser 在miniWob++任务上的训练
我们的目标：跑通：nemo-rl+nemo-gym+lexmount browser 在真实网站WebVoyager 任务上的训练

参考文章：
https://docs.browserbase.com/integrations/prime-intellect/rl-training


目标：
复现 BrowserEnv的训练
证明 lexmount browser 不弱于 BrowserBase


区别是：
- 框架，我们采用：`https://github.com/nvidia-nemo/rl`
- 模型。我们采用：`/home/wf/models/Qwen3-1.7B`
- 浏览器后端，我们采用：`lexmount browser`
LEXMOUNT_API_KEY = oc2npdoTOFTK00Ys8bzVu3LJx59aXGa3
LEXMOUNT_PROJECT_ID = 61c42286-5141-4bf0-a6aa-4ff6ff5b636b

其余保持一致：
- 过滤后的 WebVoyager 任务集，600 个真实网页导航任务
- reward 是“任务是否完成”的 LLM judge，而不是答案字符串匹配；如果 rollout 没有任何 tool call，会直接给 0.0 reward。


我们可用的机器资源：
5090上有两张卡，但是每一张没法用100%的资源，只能用原显存的70% 做训练！

---
你需要做的是：
- 登录到5090机器，工作目录：/data/wf/sxh/workspace/LexBrowserEnv/
- 下载数据（数据下载在 /data/wf/sxh/workspace）
- 整理好数据为训练
- 下载nemo-RL 框架，安装训练环境
- lexmount browser作为浏览器后端
- 准备好一切，写一个能够一键运行的启动训练的脚本！

注意事项：
- 务必保证，尽可能对齐 `https://docs.browserbase.com/integrations/prime-intellect/rl-training`。要求：训练数据是：过滤后的 WebVoyager 任务集，600 个真实网页导航任务
- 最终目标，证明：LexmountBrowser 云浏览器后端 + nemo-RL 能够在真实网站WebVoyager 任务上跑通
- 给出 reward随着训练steps增长的变化曲线。记录这个很重要，最重要。这能够证明，训练无误，因为随着训练steps，reward在涨！
- 最终目标是：
    - 一键运行的启动训练的脚本
    - 自己执行这个脚本进行训练
    - 自己训练完成，完成后，给出 reward随着训练steps增长的变化曲线。（还包括，推理步数变化、学习率变化、xxx等训练细节信息）