# 无人工 prompt 标注的内部信息读出

2026-09-28。独立研究入口 `experiments.role_free_flow`。它是实际运行的测量模块，
不是已训练或已证明提升AUROC的幻觉检测器；不替换默认检测。

只输入token序列、prompt长度、special IDs、原生A/hidden/logits和模型参数。
不读人工证据/实体/约束、source_mask、正确候选或错误span。evaluate在全token读出冻结后
才读旧局部标签。支持已保存Llama3.1-8B轨迹与现有完整A/V图，不做通用模型适配猜测。

## 已实现的量

- `features.routing_layer`：同一prompt地址集合上，当前读取与过去3步平均读取的JS；
  原prompt质量及质量加权JS另存。具体地址改变而总占比相同时仍能观察变化。
- `run.candidate_lens`：最终native top5加实际词，32层RMSNorm+unembedding竞争轨迹。
  HF hidden[32]已归一化，不能再次RMSNorm。JS是受限候选raw lens，不是全词表熵。
- `features.state_turns`：raw residual检查点16..31的跨步转向和更新；不把最终归一化
  hidden同raw residual相减。保留原熵/概率作对照。
- `messages.project_messages`：每边m=A W_O V，GQA正确展开；自动候选c为模型自身最佳非实际词。
  冻结最终RMS尺度，d=(WU[y]−WU[c])*gamma/RMS(r_final)，以d·m读各层头key的有符号作用，
  同时读MLP。这个直接分解不穿过后续网络，不是门导数或原生删除效应，不按正负判断真假。
- 完整图逐层相加保留BF16舍入残差，不虚报精确守恒；各头key保存最强4个地址及分组正/负/净量，
  原始全图不删减。prompt/history/special只是结构分区，V已有上下文化，不当作根来源归因。
- `run.calibrate`：其他source的无标签混合参考、source等权CDF，四通道固定90百分位及并集。
  百分位不是p值或正常FPR。全部token保留；事件不充当事实边界或错误预测。

## 实际测量与限制

原采样16答/4来源、1532token，1516非special，全部固定前缀重放校验记录maxlogit/maxattention=0。
这些hidden是经验证的重放采集；不能将全量17790答observer缓存说成原生成器内部状态。
已有完整A/V图只有洋葱两答，本轮全343token读出；它是同原文完整前向，与原逐步轨迹分开。
原生采样参数temperature=.7/top_p=.9，缓存logits/entropy为变换前分布。

头饰错误起点H=3.335bits vs正确局部.629；实际词对自动最佳其他词margin−1.283/+2.768。
洋葱正确局部层分歧JS=.4814，错误=.2094，反对统一“错误分歧更大”。
自动并集事件628/1516=41.42%，2个错误起点都触发，但洋葱正确起点也触发，不能上线。
source14325四次都是同一9token拒答；4题不足以拟合统一真假模型，也没有完整正常标签。

洋葱t126完整图：有支持over相对in的prompt直接净量+1.8883；无支持for相对on为−1.4548。
历史净量分别+1.9702/+.8840，因此“历史越强越错”不成立。比较的自动候选和前缀不同，
这些量不是直接事实支持，不证明信息传播的唯一因果路径。最终账本最大误差.044122 logit；
原生BF16与FP32候选读出最大logit差.124569，近零候选margin不可过度解读。

新增6个科学不变量检查通过，相关flow/entropy联合14检查通过。没有新LLM前向或新AUROC。
详细计划、结果、图和完整后续概率方案保存于共享research的
`refine-logs/role_free_flow_20260928/`；原始输出在`outputs/role_free_flow_20260928/`。

## 运行命令

从graph目录，用已安装research环境的python。输出目录应另取新名，保留历史结果。

```bash
python -m experiments.role_free_flow.run \
  --samples ../reanchor/outputs/samples_20260911_145421_235 \
  --states ../reanchor/outputs/states_samples_20260911_145421_235 \
  --output outputs/role_free_flow_new
python -m experiments.role_free_flow.messages \
  --graphs ../reanchor/outputs/attributed_samples_20260912 \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --output outputs/role_free_flow_new/messages
python -m experiments.role_free_flow.evaluate --output outputs/role_free_flow_new
python -m experiments.role_free_flow.report \
  --output outputs/role_free_flow_new --report /path/to/shared/research/report
python -m pytest tests/test_role_free_flow.py tests/test_flow_audit.py tests/test_entropy_detection.py -q
```

后续优先新增自动消息的原生下游响应和条件标签增量验证。原生下游J、Fisher响应、
多剂量自动选边及统一条件检测器在本轮文档中是方案，不得冒称已实现。
