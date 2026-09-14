## 第三轮边界复审：RoutingResidual

### 七维评分

| 维度 | 权重 | 评分 | 加权 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 9 | 2.25 |
| Contribution Quality | 25% | 6 | 1.50 |
| Frontier Leverage | 15% | 6 | 0.90 |
| Feasibility | 10% | 7 | 0.70 |
| Validation Focus | 5% | 8 | 0.40 |
| Venue Readiness | 5% | 4 | 0.20 |
| **加权总分** | **100%** |  | **7.30 / 10** |

**CALIBRATION: none**

**GAP：**本轮没有 curated exemplar，评分仍按固定 rubric。相较第二轮，四项指定闭环均已落地：主假设、主指标、配对统计和 hidden 身份现在指向同一个可证伪命题。剩余差距不再是规格或代码自相矛盾，而是尚无自然 hidden 结果，且岭路由残差能否形成研究贡献仍完全由该结果决定。

### 四项核对

1. **主门槛已正确。** 规格明确以 graph−`mass_matched_uniform` 的来源宏平均首错 midrank percentile 差为主；同 source 配对 bootstrap 的 95% 区间上界必须小于 0。no-neighbors 与 permuted-weights 只作辅助诊断，不能挽救 graph-vs-uniform 失败。支持性 within-answer AUROC 差要求区间下界大于 0，且未与首错主 claim 混写。

2. **配对实现正确。** `paired_source_difference` 先在同一 eligible source 上计算逐来源 graph−uniform 差，再直接重采样差值，而非比较两个独立区间。首错 midrank 与 within-answer AUROC 均输出点差、固定种子 1000 次 bootstrap 区间及来源数；少于两个来源不输出区间，因而不能误过门槛。首错排名越低越好、AUROC 越高越好的方向也已写入输出。

3. **hidden schema 已绑定。** hidden cache 必须提供 observer model/revision、tokenizer/revision、layer、state location 六个非空字段；完整规范 JSON 参与 hash。模型拟合检查 reference schema 一致，后续 calibration/test score 也通过 `_design` 强制与已拟合 schema 一致，模型文件保存 schema hash。文档准确保留边界：metadata 声明仍依赖采集器真实性，hash 只能保证一致与冻结。

4. **假设与读出一致。** 可识别性假设已从可事后选择的尾部阈值，收敛为预定总体上的 graph-uniform 首错 midrank 改善；支持性假设对应同来源 AUROC 差。自然标签仅在冻结评价中检验这些假设，没有回流拟合或被包装成无监督定理。

### 工程一致性

hidden 节点属性保持 float32；`--max-attribute-mib` 对所有已加载节点属性累计计数并超限失败。文档明确该参数不是 RSS 上限，CSR、解压和线性求解仍需额外内存，表述与实现一致。零长度 offset 被标为不可评价并进入覆盖报告，不再默认为正常 token。当前复审环境中 `tests/test_token_information_flow.py` 为 **26 passed**。

### Simplification Opportunities

**NONE。** 当前已经是一条主分数、一个同维主零模型和两个辅助诊断；无需再删或加模块。

### Modernization Opportunities

**NONE。** 本轮不需要新增模型或扩大实验清单。

### Drift Warning

**NONE。** 任务仍是看到 token 后的无标签定位，不是前缀预警；RoutingResidual 也未被称为幻觉概率、严格 innovation 或因果信息流。

### 未确认项

- 正式自然 roster 的 hidden 尚未采集/评分，因此主区间能否过门槛未知。
- 1024 MiB 只限制属性数组；正式 roster 的峰值 RSS 仍需在运行前按文档测量。
- schema 内容是否真实对应指定 observer 状态由采集器保证，本代码验证的是字段完整性、规范 hash 和跨缓存一致性。

### Verdict

**REVISE**

四项指定修正全部通过复核，没有新的代码冻结阻塞项。REVISE 仅表示科学有效性与 venue contribution 尚未由冻结自然 hidden 评价证明；工程测试、真实 CSR 检查和合成运行不能替代该证据，也没有被如此表述。当前版本可以作为严谨工程基线冻结，后续让预注册的 graph-vs-uniform 自然结果裁决是否保留研究贡献。
