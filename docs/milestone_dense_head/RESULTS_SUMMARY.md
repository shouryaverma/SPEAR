# SPEAR 扰动预测改进 — 结果总结（明早验收）

> 一键看结果：`~/spear_venv/bin/python final_report.py`
> 主判据（你锁定的）：doubles-only（未见 A+B 组合）pearson_delta，确定性 eval，同 seed 同 regime。

## TL;DR（诚实版）

1. **打赢了 baseline,而且是大幅打赢**:在未见双扰动上 doubles_pd **+19% ~ +32%**
   (3 seed 一致、显著、收敛成立、mse_delta/DE 全指标同向改善)。
2. **但赢的不是"因果结构"**:我实现的 latent 因果机制(IPLC:稀疏潜因子 + 因果 DAG +
   interaction-asymmetry/可识别性正则)**都不是取胜原因**——最简单的 **per-gene 加性
   MLP 残差头(dense-head,无任何因果/稀疏结构)反而最强(+32%)**,稀疏因子/DAG 只会
   略微拖累。
3. **真正被验证有效的归纳偏置**:给 perturbation 一个 **per-gene、加性聚合
   (`f(A+B)=f(A)+f(B)`)、非线性、zero-init 残差**的条件表示。这是 SAMS-VAE"additive
   mechanism shift"的精神,但**不需要**稀疏 mask、潜因子、DAG 或可识别性机制。

## 决定性数据（doubles_pd @5000, 3-seed 均值）

| 配置 | doubles_pd | vs baseline | 说明 |
|---|---|---|---|
| baseline | 0.2748 ± 0.010 | — | 原 SPEAR |
| IPLC linear-K256 | 0.3267 ± 0.015 | +18.9% | 稀疏因子+加法 decode |
| IPLC gelu-K256 | 0.3368 ± 0.014 | +22.5% | 稀疏因子+非加法 decode |
| IPLC linear-K512 | 0.3476 ± 0.006 | +26.5% | 更大容量 |
| **dense-head (最佳)** | **0.3628 ± 0.021** | **+32.0%** | **无因果/稀疏结构** |

清晰单调:**结构越少、per-gene 加性容量越直接 → 越好**。全指标一致(mse_delta 从
0.0160→0.0140,de_direction 0.735→0.770)。

## 最佳配置（在 baseline 命令上加）

```
--use_scm --scm_head_type=dense --scm_dense_hidden=1024 \
--scm_lr=1e-3 --scm_warmup_steps=500
```
机制:`perturbation_emb = mean(tok_emb) + Σ_gene MLP(tok_emb[gene])`,MLP 输出层 zero-init
(初始 no-op,逐比特等于 baseline ⇒ 最差不劣于 baseline)。

**hidden 容量确认(dense, @5000, 3-seed):** h1024 = 0.3628±0.021 (+32.0%, 均值最高) ·
h512 = 0.3578±0.002 (+30.2%, **方差极小、最稳**) · h2048/h4096 过拟合(@3000 已明显掉)。
→ **推荐 h1024(峰值)或 h512(最稳);甜点在 512–1024。**

## 为什么因果结构没赢?（已实证,不是推测）

我直接检验了"因果结构是否在非加性对上才有用"(脚本 `analyze_nonadditive.py`):用真值
`Δ(A+B)` vs `Δ(A)+Δ(B)` 的偏差度量每个 test double 的非加性程度,分高/低两组比较各模型的
per-condition pearson_delta:

| 组 | baseline | dense(纯加性) | gelu256(有交互建模) | lin512 |
|---|---|---|---|---|
| 低非加性 (n=11) | 0.343 | **0.461** | 0.399 | 0.415 |
| 高非加性/上位 (n=11) | 0.190 | **0.295** | 0.234 | 0.264 |

- **即使在真正非加性(强交互/上位)的 doubles 上,纯加性 dense-head 仍明显优于有交互建模的
  IPLC(gelu synergy + 非加性 decode)**。gelu256−dense 在两组都是约 −0.06。
- 这**排除了"因为 split 加性所以因果没体现"的辩解**——**因果/交互结构在此数据上确实无益**,
  是干净、有力的负结果。dense 在更难的高非加性组反而领先更多(+55% vs baseline)。
- 可能原因:高非加性对的"非加性残差"难以预测,IPLC 的交互建模没学到有用信号;而其额外
  约束/复杂度反而损害了对加性主效应的拟合。**开放性**:其他数据/更强交互设定或不同交互
  建模方式下结论可能不同,但**在 norman additive 上,答案是明确的:加性够了,因果多余**。

## 诚实的科学过程（我如何被自己的对照逐步纠正）

1. Round1(@3000/小K/单seed):linear 略胜 gelu → 我以为"加法 decode 是关键"。
2. Round3(收敛/大K/多seed):gelu-K256 ≈ linear-K256 → **证伪** decode 线性性关键。
3. Round4(dense 对照):无因果结构的 dense-head 最强 → **证伪**"稀疏因子/因果结构"有价值。
- 每一步都用确定性 eval + 多 seed + 同 regime 对照,结论可复现。**负结果如实保留**。

## 可诚实主张 / 不可主张

- **可主张**:(a) 一个简单 latent 残差头让 SPEAR 在未见双扰动上 +32%,稳健显著;
  (b) 验证了"per-gene 加性组合"是该任务的有效归纳偏置(SAMS-VAE 精神的极简实现);
  (c) 一套干净的对照方法学(确定性 eval + 结构/容量解耦 ablation)。
- **不可主张**:latent **causal**(DAG/稀疏因子)的价值;identifiability 的实证收益——
  本 split 上均未体现。

## 复现 / 文件

- 代码:`SPEAR-main/src/models/origin/model.py`(dense-head 与 IPLC 均 config-gated,
  no-op-at-init 已验证)、`latent_scm.py`(IPLC)、git log 每步有说明+验证。
- 编排/报告:`night_runner.py`、`final_report.py`、`night_queue.jsonl`、`night_scoreboard.csv`。
- 工程修复:确定性 eval(RNG save/restore)、并发预处理 race、OD 回收自恢复 + cron watchdog。

## 建议的下一步（供你早上定夺）

1. **锁定并汇报** dense-head +32% 作为工程改进(可直接用)。
2. 若仍想验证 **latent causal 的科学价值**:在**非加性 split**(上位/combo)上重跑
   IPLC vs dense 对照——那里因果/交互结构才可能真正胜出。
3. dense-head 天花板调优(Round5 结果)+ 是否值得更大 hidden/更深。
