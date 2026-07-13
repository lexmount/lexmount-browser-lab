# 配对日志与机制复测审计

## 结论层级

**可以支持：**在本次 LexBench 配置中，Lexmount 完整服务组合（远端浏览器、出口 IP、
地区路由、指纹和会话运行时）比 Mac 本地 Chrome 获得更高端到端成功率。

**主要机制证据：**差异集中在站点或访问环境。它不是远端 Chromium 内核“更聪明”的
证据，也不能证明换成相同出口后仍保留 13.33pp 优势。

**不能支持：**Lexmount 在所有站点都更稳、浏览器内核本身更强，或单次失败可稳定复现。

## 全量日志分解

全量 210 条中有 64 条单边成功：Lexmount-only 46，Local-only 18。

| Loser 主分类 | Lexmount-only | Local-only |
| --- | ---: | ---: |
| E1 bot defense | 15 | 3 |
| E2 access barrier | 2 | 0 |
| E3 site limitation | 14 | 2 |
| M1-M4 Agent / evidence | 15 | 10 |
| H1 Harness | 0 | 3 |

站点/访问类净不对称为 `31 - 5 = 26`，总成功净差为 `46 - 18 = 28`。非 E 类净差只有
`15 - (10 + 3) = 2`。E 分类来自同一 Judge 的失败归因，因此只能作为机制证据，
不是独立网络测量。

原始日志提供了第二层验证：

| Loser 原始信号 | Lexmount-only 的 Local loser | Local-only 的 Lexmount loser |
| --- | ---: | ---: |
| captcha / bot challenge | 12 | 2 |
| HTTP access denial | 6 | 0 |
| navigation / `net::ERR_*` | 13 | 4 |

信号可在同一任务重叠。31 个本地 E 类 loser 中 25 个至少有一项原始信号；5 个反向
E 类 loser 中 1 个有原始信号。

## 反向顺序复测

选取 12 条机制样本，并发 4，不用于重新估计总体成功率：

1. 原始顺序：Lexmount → Local。
2. Replay 1：Local `20260713_104018` → Lexmount `20260713_104850`。
3. Replay 2：Lexmount `20260713_110405` → Local `20260713_111627`。

| Task | Site | Lexmount | Local | 判断 |
| --- | --- | ---: | ---: | --- |
| 22 | Baidu Wenku | 3/3 | 2/3 | 阈值/轨迹证据敏感，不支持浏览器归因 |
| 23 | Google Scholar | 2/3 | 0/3 | 偏 Lexmount，但 Lexmount 也有一次翻转 |
| 39 | Crunchyroll | 2/3 | 0/3 | 偏 Lexmount，有轮次波动 |
| 45 | GameSpot | 2/3 | 0/3 | 偏 Lexmount，一次 Lexmount DOM watchdog 失败 |
| 59 | Vimeo | 3/3 | 1/3 | 偏 Lexmount，但 Local 可偶发成功 |
| 97 | ASOS | 3/3 | 0/3 | 稳定访问环境差异 |
| 114 | Steam | 3/3 | 2/3 | 没有稳定隔离差异 |
| 122 | Baidu | 2/3 | 2/3 | 没有稳定隔离差异 |
| 138 | Youku | 0/3 | 2/3 | 反向偏 Local |
| 180 | 58.com | 0/3 | 3/3 | 稳定反向，主要是提取完整度 |
| 246 | V2EX | 0/3 | 1/2 | 偏 Local，另 1 次 Judge 429 缺失 |
| 3008 | 3DM Mod | 3/3 | 0/3 | 稳定访问环境差异 |

机制样本合计为 Lexmount 23/36、Local 13/35 个有效判断。由于样本刻意富集原始
Lexmount-only 访问失败案例，这两个比例不能与全量成功率并列使用。

## 直接证据

Task 97 的 Local 三次均在 ASOS 收到 `Access Denied`；Lexmount 三次均进入 ASOS
真实页面并完成搜索/筛选。Replay 2 截图：

- Local denial：`273a7df3fd271c652cd280aad80e74c34dc83c680e66dd0f82f516faba248dc3`
- Lexmount page：`e0655bc2e7ce1f2fab0f7f8e7be057dc9ed24f6c380f798624ca3794233ab3b9`

Task 3008 的 Local 三次均在 3DM Mod 收到 `HTTP ERROR 403`；Lexmount 三次均进入
3DM Mod 页面并使用“最多下载”排序。Replay 2 截图：

- Local 403：`f13fbfd98bfffe43dc12c1d00b66fa07782ec822ae3a9abf26d2d648debc7dc1`
- Lexmount page：`45abe4f87d6e8dba429fe6c7ba4d4b4f50bdcdc0debbe26e02aaa7bff677b911`

截图保留在原始 run 的 `tasks/<task-id>/trajectory/`，不提交 Git；哈希用于核对原件。

## 剩余混杂

- 两个 arm 没有共享相同出口、ASN、地理位置或浏览器指纹。
- Agent 调用没有固定 seed，独立轨迹会产生不同点击与提取策略。
- Judge 使用 temperature 1.0，单轨迹单次判定；12/64 个 discordant loser 距阈值
  不超过 10 分。
- 原始全量按 Lexmount 后 Local 顺序执行；12 条复测虽交换顺序，但不能覆盖全部站点。

要隔离“浏览器实现”本身，下一版需要让 Local 与 Lexmount 共享同一出口，记录 UA、
浏览器 build、出口 IP/ASN/geo，并对每条保存轨迹重复 Judge。当前最准确的产品表述是：
**Lexmount 的完整 browser backend 在该任务集上提供了更好的站点可达性与端到端结果。**
