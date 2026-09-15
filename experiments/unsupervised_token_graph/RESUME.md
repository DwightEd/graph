# 续跑与已完成样本预览

## 继续中断的运行

从 graph 根目录使用原命令和原参数重跑：

```bash
bash experiments/unsupervised_token_graph/run_all.sh
```

默认 train/test 路径及输出不变。run_all.sh 已传入 --resume。
断点粒度是整份样本，不是单个 head：samples 下已完成的 NPZ 会读取记录并跳过计算；
中断时尚未发布最终 NPZ 的那个样本重新计算。不要删除输出目录或修改 settings.json。
保留原来的 OUTPUT、CACHE、layer/head、root_bins 和索引等设置。进度条会重新遍历文件列表，
因此从 0 开始显示并不意味着重算了已完成样本。不要同时启动两个写同一输出的运行。

## 不等待全量结束，先评价已完成样本

```bash
SPLIT=train BOOTSTRAP=0 \
  bash experiments/unsupervised_token_graph/evaluate.sh --completed-only
```

等价设置为 COMPLETED_ONLY=1。test 已经有完成样本时，将 SPLIT 改成 test。
默认读取 outputs/source_carrier_information_npz_v2/<split>/samples 下的最终 NPZ；
使用自定义输出的运行应另外设置 OUTPUT 为原来的那个 split 输出目录。

预览只读取已保存的逐头分数和身份/offsets，以及原始标注。不重新读 attention、
不构图、不跑 LLM，不创建或修改 summary.json、complete.json、settings.json 或样本 NPZ。
.partial 不进入评价。样本文件列表在读取标签之前固定，报告保存明确的 evaluated_records。
没有 summary.json/complete.json 也能预览；存在旧 summary 时以实际完成的 NPZ 为准。

结果单独保存为：

```text
outputs/source_carrier_information_npz_v2/train/evaluation_partial.json
```

普通 evaluate.sh 不带 --completed-only 时仍要求完整运行，写 evaluation.json。
BOOTSTRAP=0 只关闭置信区间计算，不改变 AUROC/AP 点估计；默认 200 次 source bootstrap。

## 标注和覆盖率

优先使用原评分 settings.json 中的 annotations；如果没有配置，使用已有的
RAGTruth response.jsonl 路径，例如 ANNOTATIONS=/实际路径/response.jsonl。
这不是要求重新生成 METADATA。NPZ 内仍需原样本身份、split、回答摘要和原始 token offsets；
缺少这些信息时不能伪造标签对应关系。数据只有 train 就不能用 SPLIT=test 评价。

输出包括七个 token 评价口径、每任务指标、正例数、覆盖率，以及真实/打乱历史端点的对照。
CLI 显示全部错误 token 的主表，其余口径保存在 JSON。只有单一类别时 AUROC/AP 为 null。
注意 coverage 是已完成样本内的分数覆盖率，不是整个任务的运行完成比例。

evaluation_scope=completed_samples_preview 表示按执行顺序完成的子集，不是随机抽样或全量成绩。
训练集预览只是诊断；测试集预览不要用来选参数、选头或翻转分数方向。部分 source bootstrap
也不能消除样本完成顺序带来的选择偏差。

## 回归测试

```bash
python -m pytest tests/test_completed_preview.py -q
```
