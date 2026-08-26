# P-Cut QA 结果（已退役）

P-Cut 使用 prompt-rooted / response-closed 两种图切割，假设 hallucination token 对 response-closed 路径更依赖，因此预注册方向是较高的 closure score 更异常。

## 烟测

```text
test token: 2,066
closure coverage: 0.9952
```

| quantity | AUROC | AUPRC |
|---|---:|---:|
| raw closure | 0.2491 | 0.0252 |
| calibrated score | 0.2589 | 0.0271 |
| absolute position | 0.7173 | 0.0628 |
| relative position | 0.6876 | 0.0565 |

Score 与绝对位置的 Spearman 相关为 `0.1157`。

## 全量 QA

```text
test token: 30,470
closure coverage: 0.9951
```

| quantity | AUROC | AUPRC |
|---|---:|---:|
| raw closure | 0.4209 | 0.0734 |
| calibrated score | 0.4210 | 0.0730 |
| absolute position | 0.6171 | 0.1129 |
| relative position | 0.6066 | 0.0949 |

Score 与绝对位置的 Spearman 相关为 `0.0638`。

## 结论

1. P-Cut 在烟测和全量实验中都没有达到预注册方向。全量结果也明显低于绝对位置和相对位置基线。
2. 条件校准不是主要失败原因；raw closure 与 calibrated score 几乎相同。
3. 全量 AUROC 为 `0.421`，事后翻转只能得到约 `0.579` 的 AUROC，而且翻转方向来自测试标签，不能作为无监督结果。
4. 烟测的强反向关系没有在全量数据中稳定复现，说明它不能被解释为一个可靠的“相反机制”。
5. 当前 rollout 传播的是人工 sinusoidal token identity，cosine change 更像边分布重排量，不能证明证据依赖。
6. row-mass renormalization、provenance 上下界饱和和 unknown fallback 都可能改变两种 cut 的尺度，但这些属于表示问题；更根本的是 closure 假设没有得到支持。
7. P-Cut 代码已从 active implementation 中删除。结果保留，防止以后通过翻转方向或换名重新包装同一方法。
