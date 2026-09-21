# 教学阅读顺序

## 1. 先看数据与位置

读 `dataset/schema.py`，再看 JSONL / RAGTruth 适配器。Example 不知道 PyTorch 或 attention。
`labels=None`、`labels=[]` 的区别必须先讲清：不知道是否错误，不等于正确。

读 `tokenization.py`：prompt 长度 P，回答 token t 在输入 query P+t−1 处预测。
生成 IDs 是事实记录；重编码只核验字符 offset，不能用来替换生成 IDs。

## 2. 看重采样，而不是预设正负样本

读 `generation.sampling_inputs → sampling_jobs → generate_run`。
同一个 prompt 的原回答先合并来源身份，再按 draw/seed 采样。
采样后先保留所有回答，审阅后才调用 pairing；不能按模型置信度自动贴“正确”标签。

## 3. 看模型执行的表征

读 `model/sites.py` 的轴表，再看 `ModelAdapter.module_at`。
Residual 是整层输入/输出；QKV 是投影结果；head_readout 是 o_proj 的输入；
MLP activation 是门控激活后的 down_proj 输入。它们不是同一种空间。

Llama/Mistral/Qwen2 共用一套明确布局，因此没有机械写三个相同的适配器类。
新增布局改变 model 的映射，不要求 dataset、capture、operations 知道模型家族。

## 4. 看采集与状态容器

读 `CaptureSpec → capture_hooks → LayerCapture → ModelState`。
一层执行完就写盘，关闭 hook；离线分析只读需要的一层。
`ModelState.at` 和 `LayerState.at` 接收绝对位置，避免 donor 替换时手算 P+t−1 产生错位。
全局 embedding/final_hidden 与逐层状态都能用同一接口访问。

## 5. 看张量操作

```python
# A: [head, query, source], V: [head, source, width]
head_output = (attention @ values).transpose(1, 0, 2)
```

等价于 `np.einsum("hts,hsd->thd", attention, values)`。
重复而未写入输出的 s 被求和；t、h、d 是输出轴。einsum 很常见，
但本项目的这些式子用矩阵乘法已经足够清晰，因此采用后者。

读 `Target.indices`：broadcast index vectors 选择多个轴的笛卡尔积，
不是把 (head0,token0)、(head1,token1) 错配成两个点。
再读四个操作，每个只有一条可解释的变换。

## 6. 区分观察与真正干预

捕获 attention 权重只需要观察 native 输出；干预权重必须先改 A，再计算 A@V。
`model.attention_forward` 保留原生 projection、RoPE、mask、GQA 和输出投影。
`intervention.py` 只安装/清理，不推断语义来源。

Delete attention sources 和减去相应 A@V 消息可以等价；
Delete 整个 head、Replace V、Inject residual、Steer MLP 则是不同实验。
四种条件仅是检验两组效应交互的实验设计，不是框架能表达的最大操作数量。

## 7. 最后才做具体审计

`analysis.vectors.compare_vectors` 返回带 group/pair 轴的结果；
`analysis.audit` 的固定列名仅为旧 reanchor 报告导出。
把后者换成一个 hidden-state 或 MLP 分析，不需要改前面任何一层。

必须先跑同状态替换/零剂量对照，再解释其他干预效果。
attention 变化、logp 变化、范数/余弦变化都不自动等于幻觉机制；
正常/错误对照、标签连续性和来源适用性仍须单独检验。
