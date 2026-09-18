# 导出实际参数与配对输入

在 graph 根目录运行：

```bash
python -m experiments.charm_structure_audit.main --mode export_circuit
```

默认读取 `outputs/charm_structure_audit_qa/QA/seed_0/audit_whitebox/`。
输出 `outputs/charm_structure_audit_qa/QA/seed_0/audit_circuit_export/charm_circuit_inputs.tar.gz`。
`--root` 指定其他 seed 目录，`--output` 指定其他导出目录。

代码只做三步：读现有清单 → 提取七个数组 → 打包。
checkpoint 使用 whitebox 清单中记录的实际路径，直接复制，不反序列化、不运行网络。
不读取图边，不重新计算积分，不挑选高分 token，不改变旧分数、标签或实验。
缺少清单列出的文件时直接抛出文件系统错误，不跳过或补数据。
压缩成功后才替换输出文件；中断留下 `.partial`，不会当成完整结果。

压缩包内容：

```text
manifest.json               原配对清单与来源信息
geometry.json               原 LLM 层数、头数
node_only/checkpoint.pt      实际训练参数
inputs/*.npz                error_x / normal_x / logits / baseline_score /
                            tokens / text / threshold
```

这为后续核验 `head组合 → MLP激活 → logit贡献` 提供必要输入；
导出本身不产生新的实验成绩或参数功能结论。不要把模型文件提交到 GitHub。
