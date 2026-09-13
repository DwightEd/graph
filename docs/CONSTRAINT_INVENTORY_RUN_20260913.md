# 完整来源图：CPU构造与坐标核验

前置：`constraint_inventory`独立工程Required关闭。沿用research@03909e02环境，无包/权重变化，无安装。此阶段只修候选表示，不运行模型或读取幻觉标签，不测试“关系是否影响输出”。完整来源保留未知值、长文本、字符串内词/值组件、field/context/record包含关系；七天key bundle仅代表实际源字段，不代表query all-days量词已识别。

独立fresh agent一次执行：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.constraint_inventory --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/constraint_inventory_v1_20260913
```

不要覆盖/重试已有目录。期望17790 response refs、2965去重sources，以实际冻结输入为准；字段/组件/上下文/包含边/unknown/longtext/精确mapping unavailable分母必须完整记录。Unsupported literal spelling仍保留全字段，不能伪造组件源坐标。

核验settings/input/code/live+snapshots/source artifacts/manifest，逐source所有保留raw span确实来自原source，response refs的source_span确实对应完整prompt。不把raw source覆盖率当semantic fact completeness。source13717/14637的旧长评论和unknown字段都应保留，地址220/套间1应成为address field的子组件；weekday inventory完整7members但query_requirement_verified必须false。记录实际session/exit与耗时，报告 `refine-logs/constraint_inventory_doc_witness_20260913.md`。

此后ranker需先明确自动query表示和真实训练视图，不能使用假定正确的relation/owner/condition token桶或将同句槽位强制同事件。没有新GPU授权门禁；只是不执行尚未设计好的模型。
