# ICLR 项目代码与实验规则

这份规则来自历次代码审计和实验讨论。它不是建议，而是当前项目的默认约束。后续新增或修改代码前，先按文末清单逐项检查。

## 1. 仓库和版本

- 只维护一个正式实现，不再出现 `v2`、`new`、`final` 等并行版本。
- 当前正式实验入口是 `experiments/directed_route_hypergraph/run_qa.sh`；
  GroundedRoute、Information Flow 与 HoloRoute 只作为基线、控制或历史记录。
- 正式代码直接进入 `main`；不交付 patch，不依赖 `gh`，不留下长期实验分支。
- 旧方法如果已被结果否定，只保留结果、结论和必要的复现实验说明，删除重复代码。
- 一个概念只保留一个入口。例如 detector、evaluation、pipeline 不同时存在两套同义文件。

## 2. 代码组织

- 模块按研究步骤拆分，而不是按“工具函数集合”拆分。
- 每个文件只负责一件事；主流程应当能按文件名顺序读懂。
- 核心函数使用直接的动词命名，例如 `build_graph`、`route_moments`、`score_detectors`。核心逻辑不使用以下划线开头的名字隐藏起来。
- 不返回包含几十个命名字段的字典，不把大量手工特征逐个赋值后再拼接。
- dataclass 只描述真正的数据对象，例如图、节点表征和配置；不为每个中间结果创建一层包装。
- 研究代码优先可读，避免为极少发生的情况加入多层 `if`、重复 `raise`、文件哈希链和 schema 防线。
- 必要检查只放在外部边界：读取数据、加载用户指定文件、保存最终 artifact。模块内部依赖清晰的数据接口，不重复验证同一件事。

## 3. 构图和节点表征

- 一条 prompt-response 样本对应一张图。
- 节点是 token；边保存准确的 source、target、layer、head 和 attention weight。
- 不在构图前平均 layer 或 head，不把 32×32 结构过早压成一个标量。
- 未保存的稀疏 attention 是“低于阈值”，不是观测到的零；使用 `unresolved` 质量表示。
- 邻居状态和边属性必须在图编码阶段聚合进节点。最终 detector 只读取 `node_embedding`，不再运行第二个 GNN。
- 聚合不能只做未归一化求和。至少要区分注意力质量与邻居内容，并避免节点度数直接决定向量尺度。
- 当前正式聚合将 `(target,layer,head)` 作为有向 row hyperedge，使用
  P/R-conditioned slots、head pooling 和 layer-ordered GRU 更新节点。
- ordered endpoint target 可以在形成单层 transition 时固定平均 heads，
  但原始边、local row loss 和 encoder message passing 必须继续保留 head identity；
  该 target 只能称为 attention transport proxy。
- unresolved sink 与 self endpoint 必须和 non-self endpoint shape 分开建模，
  降低深层 rollout 只学习缺失质量或 self identity 的捷径；位置/长度仍需独立控制。
- 精确 endpoint、edge weight 与 endpoint 的配对、以及邻居消息是否有用，必须通过独立构图控制实验验证。

## 4. 无监督与标签边界

- 主方法保持无监督。训练图编码器和异常检测器时不读取 hallucination label。
- 标签只用于两处：最终后验评价；单独标注为 diagnostic 的监督 readability probe。
- 监督 probe 不能替代无监督结果，也不能用于选择主方法的方向或超参数。
- 划分以 `source_id` 为组，不能让同一 source 同时出现在训练和测试。
- 位置、回答长度等偏差必须单独报告，不能混入节点特征后再把结果解释为图信号。

## 5. 实验设计

每个实验必须记录：

```text
问题与假设
使用的数据和样本范围
具体代码模块
运行命令
提交 SHA
输出目录
指标与置信区间
位置/长度对照
结论：保留、修改或停止
```

- 不把烟测结果当正式结论。
- 不在看过 test label 后翻转分数方向、选择最佳 head、选择最佳 detector 或重新定义机制。
- 图构造对照必须独立训练和编码，不能只在固定 embedding 旁边事后重连边。
- 对照至少包含：`no_message`、`endpoint_rewire`、`weight_shuffle`。
- 相同 detector、相同训练预算、相同 seed 和相同 token 行用于成对比较。
- 无效结果同样写入实验记录，防止以后换名重复运行。
- 已失败的 P-Cut closure 不得通过新 encoder、翻转方向或改名重新成为主分数。

## 6. Shell 与运行入口

- 每个正式实验提供一个一键 `.sh`。
- 脚本保持直白，不使用 `set -euo pipefail`。
- 每个阶段直接运行命令，并用 `|| exit $?` 保留完整 traceback。
- 数据路径、输出路径和主要参数集中写在脚本开头。
- 不在脚本中自动跳转、打开 JSON 或隐藏错误。

## 7. 测试

- 测试核心数学与数据变换：质量守恒、因果方向、节点对齐、聚合结果、控制实验实际改变的对象。
- 不把测试主要写成文件哈希和人为 schema 检查。
- 每个新模块至少有一个能说明研究含义的测试，而不是只测试 import。
- 合成测试通过只代表实现符合预期，不代表方法在真实数据上有效。

## 8. 写作和注释

- 注释解释“为什么这样计算”，不要逐行复述代码。
- 文档使用普通语言，先说明问题，再写公式和代码位置。
- 不用 `holonomy`、`causal`、`grounding` 等词包装尚未验证的统计关系。
- attention-only 阶段只能声称路由结构，不能把 attention weight 直接称为功能贡献。

## 提交前检查

```text
[ ] 是否只有一个正式入口？
[ ] 文件名能否直接对应研究步骤？
[ ] 核心函数是否短、清楚、没有不必要的防御分支？
[ ] layer/head 和 exact endpoint 是否被保留？
[ ] ordered target 是否对比 reverse/last-layer，且没有被 sink/self 支配？
[ ] 邻居信息是否已经进入 node_embedding？
[ ] detector 是否只读取 node_embedding？
[ ] 是否提供 source-disjoint、位置基线和构图控制？
[ ] 是否有一键脚本且不使用 set -euo pipefail？
[ ] 实验假设、命令、结果和停止结论是否已记录？
```
