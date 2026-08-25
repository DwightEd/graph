# HoloRoute-Base QA 结果（已退役）

这份结果来自旧的 attention-event graph masked reconstruction 方法。结果保留用于对照；对应神经编码、随机遮蔽和六残差联合检测代码已经从当前实现中删除。

## 数据范围

```text
任务：QA
测试 token：30,619
方法：旧 HoloRoute event/depth/relay/query/holonomy residual
```

## 各残差

| residual | tokens | coverage | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| event | 30,619 | 1.0000 | 0.6120 | 0.1337 |
| depth | 30,619 | 1.0000 | 0.6275 | 0.1265 |
| relay | 30,470 | 0.9951 | 0.6014 | 0.1035 |
| query | 30,619 | 1.0000 | 0.6007 | 0.1225 |
| depth-relay disagreement | 30,470 | 0.9951 | 0.5972 | 0.1067 |
| holonomy | 30,470 | 0.9951 | 0.4763 | 0.0683 |

## 最终分数与位置基线

| score | Spearman with absolute position | AUROC | AUPRC |
|---|---:|---:|---:|
| HoloRoute final score | -0.0500 | 0.5956 | 0.1029 |
| absolute position | 1.0000 | 0.6171 | 0.1129 |
| relative position | 0.8766 | 0.6066 | 0.0949 |

## 结论

1. 最终无监督分数没有超过绝对位置基线，不能作为有效 detector。
2. `event` 和 `depth` residual 含有少量标签后验区分信息，但联合校准后没有形成有效预测。
3. `holonomy` 低于随机方向，当前 curvature 叙事没有实证支持。
4. 结果再次说明“正常 token 更易恢复、幻觉更难恢复”不是可靠的核心假设。
5. 旧 HoloRoute 只保留为 Git 历史和本文件中的基线结果；当前代码已经覆盖为 P-Cut。
