# P-Cut QA 烟测结果（已退役）

P-Cut 使用 prompt-rooted / response-closed 两种图切割，假设 hallucination token 对 response-closed 路径更依赖，因此预注册方向是较高的 closure score 更异常。

## 数据范围

```text
test token: 2,066
closure coverage: 0.9952
```

## 结果

| quantity | AUROC | AUPRC |
|---|---:|---:|
| raw closure | 0.2491 | 0.0252 |
| calibrated score | 0.2589 | 0.0271 |
| absolute position | 0.7173 | 0.0628 |
| relative position | 0.6876 | 0.0565 |

Score 与绝对位置的 Spearman 相关为 `0.1157`。

## 结论

1. P-Cut 没有达到预注册方向。`AUROC < 0.5` 不是纯随机，而是关系方向明显相反；事后把分数乘以 `-1` 会得到约 `0.751` 的 AUROC，但这属于标签后验翻转，不能当作无监督结果。
2. 条件上尾校准没有造成主要失败；raw closure 与最终 score 的方向基本一致。失败首先来自 closure 假设或切割表示本身。
3. 当前 rollout 传播的是人工 sinusoidal token identity，cosine change 更像“边分布重排量”，不能证明证据依赖。
4. row-mass renormalization、provenance 上下界饱和和 unknown fallback 都可能改变两种 cut 的尺度。
5. P-Cut 代码从 active implementation 中删除。结果保留，防止以后只通过翻转方向重新包装同一方法。
