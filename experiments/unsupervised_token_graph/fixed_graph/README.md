# 固定图表示的无监督验证

**没有GNN训练、重构损失、来源错接分类器或大模型前向。**
复用现有attention，以固定图变换生成表示，在无标签参考来源中计算近邻距离。
所有对照一次提取、同一参照采样、同一评分算法、同一覆盖范围比较。

## 一次运行

在 `graph` 根目录、`research` 环境：

```bash
python -u -m experiments.unsupervised_token_graph.fixed_graph \
  --phase all --output outputs/fixed_graph_v1 --threads 4 --resume
```

默认原路径：

```
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset
```

`--train-cache`、`--test-cache`、`--dataset` 可显式指定；`--index`复用已有inputs/records。
默认所有任务与生成器；`--tasks QA Summary Data2txt`、`--generators llama-2-7b-chat`可过滤。
裸六字段缓存从原response/source_info补齐身份，只有身份白名单进入拟合，不用labels。
元数据解析会打开含标签字段的JSON，但标签值不进入表示、参照采样或校准。

先查看接口：`--phase inspect`。只评价已有分数：`--phase evaluate --output ...`。
没有offset时沿用原token-ID精确核验；只在评价阶段需要原本地observer tokenizer：

```bash
python -m experiments.unsupervised_token_graph.fixed_graph \
  --phase evaluate --output outputs/fixed_graph_v1 \
  --tokenizer /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
```

也支持此前修好的TOKENIZER环境变量和已知本地目录查找。不加载模型权重。
这是CPU实现，线程数显式控制；没有--device/--epochs，因为没有神经网络训练。
不使用set -e、不后台。首个token缺预测行时保留NaN，报告覆盖，绝不移动标签填洞。

## 核心算法与先验

对每个物理layer/head计算九维节点属性X：prompt/local/remote质量、整行/local/prompt
attention熵、local最大份额、当前词的历史复现质量、prompt最大份额。
熵以可用位置数量的对数归一化；它不是logits熵。缺少hidden/logits时不造向量或分数。

PL、PR分别保留local/remote历史边的原权重；不因删掉缺测邻居而放大余下边。
每个head分别得到：

```
当前          X
局部继承      PL X
远处继承      PR X
当前与继承差  X - PL X - PR X
局部两跳      PL PL X
局部后远处    PL PR X
远处后局部    PR PL X
后文读取      PF X
```

PF是原历史图转置后的行归一化分析算子，只用于读取后文，不改变原模型因果方向。
原节点以“预测回答词t”的位置q=P+t-1对齐，原key对应已经输入的历史词。
同head的连乘是检测图上的结构滤波，不是原Transformer同层两跳运算，更不是已解码的概念链。

这些分支让“当前分散”与“低熵续写读到怎样的历史”具有不同表示。
不要求首错一定高熵，不根据金标建立起点，不用回看门强制重置历史。
变化分支保留候选重新选择的线索，但并不声称已经自动找到了真正reanchor。

不同head的分量用物理layer/head/分支编号确定的固定双哈希CountSketch压缩到128维。
这是有损随机投影，不是先平均head；投影种子固定，不读标签、不搜索最佳seed。
每个对照使用相同输出维数，不能把压缩后的向量叫完整、可逆的逐头语义表示。

## 同时输出的六个方法

| 方法 | 只改变什么 |
|---|---|
| nodes | 仅X，同样的固定投影和近邻评分 |
| graph（主方法） | 全部八个关系分支 |
| shuffled | 相同X与算子形式，换历史边的具体端点 |
| one_hop | 不含三个二跳分支，保留后文分支 |
| no_future | 不含PF分支，保留二跳 |
| node_smooth | nodes的逐词近邻分数只做时间滑动平均，单独校准 |

shuffled在同head、query、距离带、local/remote、token复现类别和可观测类别内置换。
距离带为1、2、3–4、5–8等，包含零权重槽位；保留各组质量、非零数和权重多重集。
不把精确距离保持不变说成已经实现。原prompt边不打乱。
参照和测试均使用同一打乱规则，不用原图参照去评价一个人为损坏的测试图。
实际改变强度保存为0.5*sum(abs(A_shuffled-A))；均匀权重的空扰动正确记录为0。

原始逐头计算不混合；仅local_mass和attention_entropy这两个独立基础对照跨head求均值。
后文分支为离线信息；no_future只移除此分支，不是经过认证的在线系统。
所有已观测token都评分，超过校准阈值的连续位置成段；允许单词、长段和整答无报警。
local-window=16仅决定边类型及纯平滑半径，不限制错误片段长度。

## 无标签参照与校准

官方train按source分为80%参照来源、20%校准来源，官方test不参加任何拟合。
同source所有回答必须在同一分区。各task/generator单独估计参照，不能将不同生成器当同分布。
每个参照来源最多16个随机token，默认参照库最多4096行，所有方法使用完全相同的行。
中位数/IQR标准化也仅拟合参照库。分数为标准化表示到5个最近参考token的平均距离。
这一步是统计估计，不是神经网络训练；没有筛选“正确回答”建立normal库。

校准来源全部可观测token用于按source等权确定分数位置、尺度与95%分位阈值。
这个阈值是混合无标签数据的异常预算，不是保证5%正常误报的阈值，也不强制test报5%。
分数越高表示结构越少见，不是事实错误后验。特征设计参考已有标签辅助发现，
但代码拟合不用标签；反复研究过的test仍应称探索性测试而不是全新确认数据。

## 代码与输出

```
inputs.py      原缓存与身份接口
operators.py   原属性、固定图变换、匹配打乱
reference.py   source拆分、近邻距离、无标签校准
pipeline.py    提取 → 建参照 → 保存分数
run.py         一个命令按阶段执行
 evaluation.py 冻结后才读金标；复用原七种token口径
```

```
outputs/fixed_graph_v1/
  settings.json
  features/manifest.json, inputs.json, *.npz
  reference/settings.json, complete.json, <group>/bank.npz
  predictions/*.npz, freeze.json, evaluation.json
```

每答只存五种128维表示及分数，不复制全量高维hidden或1024张稠密attention。
逐head流式读取；实际保留质量、缺测邻居质量、打乱强度保存在diagnostics。
其列顺序为layer、head、改变质量、历史质量、缺测邻居质量、平均原行质量。
原缓存不做新增top-k截断；其原有稀疏截断无法恢复。
打乱索引单head需要O(R²)整数空间，R是回答长度，不创建[L,H,N,N]；极长回答应先检查内存。

prepare逐答续跑，score逐答续跑；fit没有epoch，中断会重建小型参照，不重提attention。
输入attention修改或表示/参照参数改变必须新建output，避免混用结果。
原metadata/offset读取、span_audit、CHARM及其他实验不改；旧无依据的错接训练不恢复。

## 先看什么结果

`evaluation.json`：每任务/生成器及ALL的七种token口径、AUROC/AP、覆盖，
片段精确/IoU匹配、边界偏差、结束后8个正常token误报、正常回答报警率。
`graph_minus_control`按source配对重采样，在全错误、首错、延续和严格首错后分别报告增量。
所有对照共同覆盖；不会挑一个最好的头/变换再把它当注册主方法。

图胜nodes：固定邻域信息有增量；胜shuffled：具体端点有增量；
胜node_smooth：不只是平滑；胜one_hop/no_future：相应分支值得保留。
哪一项没有增量就据实报告，不能只展示原始AUROC或把零扰动当反证。

局限：这里的prompt属性保留分布形状，不保留完整事实身份，也不能核验年份/阶段的适用性。
监督可分不保证样本在无标签密度中异常；常见错误、罕见正确仍可能混淆。
本次验证的就是已知结构先验能否用简单固定表示读出，不承诺效果，亦不假称完整图散射理论保证。

本地测试命令：

```bash
python -m pytest experiments/unsupervised_token_graph/fixed_graph/tests -q
```

实际本地结果：24项新测试及33项span_audit回归通过，未运行服务器2497/449条自然数据。
