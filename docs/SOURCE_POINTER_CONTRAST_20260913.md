> 更新：完整自然A/B准备已执行，36回答、426claim、1704尝试、10个未验证候选；1394 whole-event role、222 subject、54 response owner、23 source owner、1 quoted 拒绝。候选存在语法与范围错误，不能声称成功。22项CPU检查和独立C0/R0只证明接口完整性。输出 outputs/source_pointer_bridge_prepare_20260913，drafts SHA 170f99f2dcfd4b0192892285847ffe3cf3f8d5b4409b43a351e114a1d85578。该版本4文件随prepare冻结；下一步以新surface模块删除SRL硬前置。下文原实现记录保留。

# 来源节点驱动的正误事件对照：下一版接口

本模块位于 `next_iteration/`，没有改变运行中的 soft_graph_v1 的47份代码。它补的是严格机制路径的对照供给，不是另一个幻觉分类器。当前已完成CPU实现与21项检查，独立复审进行中；尚无自然样本验证或native结果。

## 模型中的位置

现有 A 原文指针/字段图 → B 高维事件/角色匹配 → **固定来源节点与回答角色做单槽位替换** → 四项有限验证 → 原有 `CausalOracle` 上的完整事件差。

来源节点的唯一身份包含具体 occurrence 和 event/record membership；同值不同记录分别保留。替代值只来自该节点的原文片段，或字段图中安全解码的字符串/原始数值。没有开放QA生成的答案，没有RAGTruth标签输入。保留每个既有B候选的尝试与拒绝，不补搜更容易成功的候选。

有限验证同时要求：原事件 C+N≥0.8、改后事件 S≥0.8、选中来源节点在其实际owner/条件下支持改后角色≥0.8，以及非目标条件/归属/语法保留≥0.8。它们是冻结reader的条件预测，不是真值。完整reader请求缓存、模型/tokenizer/code身份及原始预测都绑定到对照；最终token化前再次验证。

## 精确接口

`source_pointer_contrast.prepare_contrast(row, A, B, claim_id, term_index)`：重编译两端图，检查B节点、来源membership、角色span和事件成员关系；按确定性surface策略生成原/改文本或显式拒绝。

`verify_contrast(reader, draft, reader_identity=...)`：reader使用 `reader_receipt.ReceiptReader`，只执行四个有限选项请求。请求看完整source、原事件/改事件和此前response；此前response仅用于解引用。结果引用已经发布的不可变缓存文件。

`finalize_contrast(row, A, B, draft, verification, tokenizer, reader_identity=...)`：重新构造draft核对一致性，使用现有 `build_contrast` 和 `slot_masks` 生成两个互不重叠的完整token事件、共享前缀、source keys和目标slot masks。既有96个continuation token上限仍生效；超限不得截短事件。输出 `certificate_count=0`，随后才有资格进入native测量。

正误目标是 `F=log P(原事件 B)-log P(来源支持的替代事件 A)`，两分支共享原始prompt与编辑前response。概率差覆盖完整事件，不取第一个不同token代替；同时保留slot/context分解。

## 当前允许与拒绝范围

允许 object、quantity、time、location、duration、attribute、origin、destination。数值使用原始literal，不推断单位、做换算或把时间范围算成持续时长。非目标字符严格不变；语义保持还需有限验证。

拒绝 subject、predicate、condition、negation、bool、None、带引号/code的目标、空字符串、不可打印值和完全相同的surface。主语改变可能连带指代/一致性，多角色错误可能形成混合事件；本版不偷偷扩展成自由重写。改后句子被全文支持但选中记录未获支持时，仍不能给该记录归因。

## 准备全部自然候选（CPU）

工作目录 `graph`，待父批所有36回答的A、B完成后：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.source_pointer_prepare --run outputs/soft_graph_v1_20260913 --output outputs/source_pointer_bridge_prepare_20260913
```

该命令尚未执行。先检查输出目录不存在；不覆盖任何先前结果。它核验父批全部执行代码、A→B artifact和B高维文件，执行前保存本模块4个代码文件快照；执行前后与发布前检查哈希。写出全体drafts及拒绝分母，manifest最后发布。GPU调用0、reader调用0、标签读取0。准备成功不是有限验证通过。

## Native后续与未解决问题

最小native复用方案见 `refine-logs/source_pointer_native_bridge_spec_20260913.md`。在同一完整F上进行全层、全共享前缀的query组搜索，冻结位置与原始输入两类对照，再做有限关系过滤、位置/来源验证、条件pairedR及证据×MLP四状态交互。两类对照的估计目标不同，不能混用soft_graph_v1各节点到各自V的对照。

raw-input→selected-V→recipient路径通过，只证明输入来源与所测路径。它仍可能传递答案内容，不能直接称abstract binding pointer。回看可分布在联合query组；最大效应点不是唯一必需节点，RAGTruth也没有内部节点真值。

自然效果、严格证书覆盖、内部独立识别适用约束、路由/采纳失败分类仍未验证。另已发现soft_graph_v1边界会受固定地址单元和不完整anchor影响，且claim分数扩展到每词可能扩大错误范围；下一版分段与词级scope需独立修正，不能把这次对照供给器当作已解决全部问题。

## 依据

设计与边界复审：`refine-logs/source_pointer_contrast_bridge_review_20260913.md`。工程审查：`refine-logs/source_pointer_contrast_engineering_20260913.md`。指针/内容区别的原始方法与本项目推论：`refine-logs/binding_pointer_payload_methods_20260913.md`。
