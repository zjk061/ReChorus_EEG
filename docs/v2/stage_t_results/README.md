# 阶段 T 结果摘要

阶段 T 已按 `docs/v2/阶段T执行计划.md` 完成有限训练优化筛选。结论是：没有任何训练优化候选同时满足 GAUC 均值、逐 fold/seed 稳定性、配对用户簇 bootstrap 置信区间和校准约束；因此 T5 五种子确认被短路，最终保留 `H2-history-behavior`。

## 最终决策

- split：`protocol_a_rollv2_cv3`
- locked test：未访问，`locked_test_accessed=false`
- 正式筛选运行数：81
- T5：未执行，原因是 T2/T3/T4 没有候选通过全部筛选门槛
- final action：`keep_H2_history_behavior`

## 汇总指标

| 阶段 | 配置 | runs | GAUC mean | GAUC std | AUC mean | LogLoss | Brier | ECE | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T2 | `sch-none_bal-sample_bce_pair-0p0` | 9 | 0.612138 | 0.022860 | 0.625080 | 0.635543 | 0.216409 | 0.065623 | 保留 |
| T2 | `sch-plateau_bal-sample_bce_pair-0p0` | 9 | 0.604400 | 0.016342 | 0.616693 | 0.639409 | 0.218272 | 0.066874 | 未通过 |
| T3 | `sch-none_bal-inverse_user_count_bce_pair-0p0` | 9 | 0.612004 | 0.020030 | 0.629195 | 0.636179 | 0.216177 | 0.063242 | 未通过 |
| T3 | `sch-none_bal-user_balanced_sampler_pair-0p0` | 9 | 0.620198 | 0.026770 | 0.635423 | 0.628959 | 0.214257 | 0.056292 | 均值较高但稳定性/CI 未通过 |
| T4 | `sch-none_bal-sample_bce_pair-0p05` | 9 | 0.609363 | 0.017198 | 0.620076 | 0.638143 | 0.218056 | 0.067610 | 未通过 |
| T4 | `sch-none_bal-sample_bce_pair-0p1` | 9 | 0.612901 | 0.020230 | 0.619203 | 0.638694 | 0.218284 | 0.070802 | 未通过 |
| T4 | `sch-none_bal-sample_bce_pair-0p2` | 9 | 0.611706 | 0.018342 | 0.620127 | 0.639241 | 0.218324 | 0.072395 | 未通过 |

## 关键门槛结果

- T2 Plateau scheduler：GAUC 均值低于 no scheduler；配对 ΔGAUC 95% CI 为 `[-0.039328, -0.003352]`，未通过。
- T3 inverse-user BCE：GAUC 均值没有超过普通 sample BCE；配对 ΔGAUC 95% CI 为 `[-0.043507, 0.006503]`，未通过。
- T3 user-balanced sampler：GAUC 均值提高到 `0.620198`，但 fold1 只有 1/3 seeds 不低于基线，且配对 ΔGAUC 95% CI 为 `[-0.008780, 0.029099]`，跨 0，未通过。
- T4 pairwise：
  - `lambda=0.05`：配对 ΔGAUC 95% CI 为 `[-0.046941, -0.009328]`，未通过。
  - `lambda=0.1`：均值略高，但 fold1 稳定性不足、配对 ΔGAUC 95% CI 为 `[-0.028451, -0.004300]`，且 ECE 相对恶化超过 5%，未通过。
  - `lambda=0.2`：fold3 稳定性不足、配对 ΔGAUC 95% CI 为 `[-0.055591, -0.002894]`，且 ECE 相对恶化超过 5%，未通过。

## 产物

- `summary.csv`：阶段/配置级聚合指标。
- `seed_results.csv`：每个 fold、seed、配置的指标。
- `decision.json`：机器可读决策、门槛、bootstrap CI、校准判定。
- `ensemble_predictions/`：按阶段、fold、配置和 EEG 对照类型聚合的预测文件。
- `log/v2/stage_t/<experiment_id>/`：每次正式运行的 checkpoint、config、metrics、prediction、per-user metrics、control predictions、corrections、training history、train log 和 environment。

