# 同题采样的原 LLM 路径冲突审计

当前定点实验、公式与限制见 [FOCUSED.md](FOCUSED.md)。原批量扫描的完整说明保留在 Git 提交 `fd8b7f7`，历史结果不改。

## 运行

```bash
# 先核查原有16条采样；不加载原模型
python -u -m experiments.path_conflict.main --study focused --stage inventory

# 定点检验四个head、来源角色、联合干预和下游恢复
python -u -m experiments.path_conflict.main --study focused

# 只读取完成的结果重新生成报告
python -u -m experiments.path_conflict.main --study focused --stage report
```

默认数据仍是 `reanchor/outputs/samples_20260911_145421_235`，原模型路径读自其settings.json。
输入必须有原始采样NPZ；只有CSV摘要不能复原实际V写入。迁移路径用`--samples`、`--model`明确指定。

新结果放在 `outputs/same_question_path_conflict_focused_v2/`，每种条件保存NPZ，可续跑。
完成后上传 `path_conflict_review.tar.gz`。`source_routes.html`可逐头查看最近读取的来源地址。

原批量扫描仍可运行：

```bash
python -u -m experiments.path_conflict.main --study coarse
```

由于首词概率改为同一次前向，重算结果放在独立的`same_question_path_conflict_coarse_v2`。
原报告可通过`--stage report --output outputs/same_question_path_conflict`重新汇总，原NPZ不会被重算覆盖。

## 阅读代码

`operators.py`是A/V/W_O消息与候选读出；`native.py`在原模型中切断或恢复写入；`scoring.py`统一首词和后续词指标。
`focused_inputs.py`明确来源角色与候选；`focused_plan.py`列出预定对照；`focused.py`运行；`focused_report.py`汇总。
原`data.py`、`cases.json`与批量报告仍保留。没有引入检测器训练、金标构图或新的幻觉分类器。

37项本地小型因果GQA测试通过，1项Transformers集成测试因缺包跳过，见`FOCUSED_TEST_RESULTS.txt`。
自然8B的本轮定点干预尚未在本地执行。局部读出支持、最终干预效果、语义因果解释必须分别报告。
