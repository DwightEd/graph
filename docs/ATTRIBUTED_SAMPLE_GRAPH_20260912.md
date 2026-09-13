# 每个prompt–response样本的属性图

2026-09-12，按用户最新澄清：节点描述高维信息，拓扑描述信息关系。当前先检验
真实正误决策窗口的约束归属可辨性；分段只分组，不截断其他历史或跨span连接。

## 表示契约

`G=(V, X, E, R)`。V为整个已观察prompt与response的token位置。
X保留每层原始4096维残差、attention后残差、实际MLP更新和V投影，不预先压缩成
少数候选方向/异常标量。E的属性为A[layer,head,receiver,sender]，消息方向
sender→receiver；完整保留prompt内部、prompt→response和过去response→后续
response的连接。R为位置、prompt/source/history/special角色及层/头身份。
角色是输入格式元数据，不是实体关系真值或正误标签。

一层/头的attention只是某种读取关系，不直接解释为语义事实关系。后续约束归属
必须联合节点内容与连接判断。节点的跨层张量保留各阶段，另可展开为(层,位置,阶段)
计算DAG，用残差和MLP连接核查路径；不能在平面token图上混用跨层边伪造传播顺序。

存档具体张量：

| 数组 | 形状 | 含义 |
|---|---|---|
| residual | [L+1,N,4096] | 每层前残差及最终层后、最终norm前残差 |
| post_attention | [L,N,4096] | 实际attention加入之后，MLP之前 |
| mlp_update | [L,N,4096] | 实际MLP输出；与前项相加得到下一层残差（含计算dtype舍入） |
| values | [L,N,KV_heads,head_dim] | 实际V投影；结合模型O投影可重构向量消息 |
| attention | [L,H,N,N] | 全部原生有向边、各层各头分别保存 |
| roles/input_ids | [N] | 节点元数据，不含正确性或约束归属标签 |
| target_ids/top_ids/top_logits/entropy | [T,...] | 保存下一词对齐和原生输出对照，不替代X |

Llama-3.1-8B，L=32、H=32。bf16实际值转float32保存，避免float16把很小的边权
下溢成0。不阈值化、不取top-k边、不平均层/头；原生有限精度本就为0的边不捏造。
输出O投影按相同模型文件身份引用，不在每个样本重复存整套权重。

N=P+T−1。预测响应y[t]使用node P+t−1，最后一个目标token不是输入节点。
完整回答的图可用于离线观察；在线读出只访问当前可见节点及其入边，不使用未来
token特征或未来反向聚合。所谓保留窗口外信息，是保留全部可见上下文。

## 本次捕获范围和检查

真实同source14375的00012/00013：前者给后续洋葱步骤补10–12分钟，后者生成
continue cooking the onions over low heat而无时长。来源和完整采样文本、旧状态
均存在；归属核对见 REAL_WINDOW_CONSTRAINT_INVENTORY_20260912.md。

旧文件attention只覆盖回答query，缺prompt内部边，且HF最终hidden是norm后状态。
因此重放各完整保存序列，在同一forward捕获全部X/E，不将旧缓存state与新attention
混拼。此次没有采样新答案，也没有给图输入实体关系或正误标签。

机制测试：小型实际Llama的原始残差/最终归一化对齐、source内部边存在、追加未来
不改变历史节点/边、目标偏一位检查、未来边拒绝。实际图逐层检查finite、因果性、
attention行和及MLP残差相加误差。通过这些仅说明表示准确存档，不证明归属可辨。

原始大数组用npy可映射读取，避免每次读出加载整图。两条样本约数GB；不构建全体
样本跨样本邻接图，也不做大规模捕获。主实现在route_graph/sample_graph.py；
reanchor/src/decoding/sample_graph_capture.py只负责机制样本加载、运行与存档。

## 归属问题的最小检验应如何读这个图

先区分表征是否含可辨信息与模型是否真正采纳。第一层比较仅X、仅E以及X+E；
第二层保持节点与边属性分布，改变连接端点或约束对应，检查读出是否随关系变化。
按语义决策对齐00012/00013，不用相同token索引强行配对，不把人工窗口当自动回看点。

洋葱beer继续烹饪没有唯一正确数值，不能把butter中炒洋葱的1–2当真值候选。评价
应区分“当前对象/动作支持的约束”与“其他对象/动作的约束”和“未给出”，而不是
数字二分类。只有这组样本无法训练和验证高维probe的泛化；拟合二者不构成信息
足够的可靠证据。需要更多独立来源的真实正误窗口或留模板/留来源的机制对照。

高维节点+拓扑的输入表示不需要标签；如果下一步训练监督诊断probe，必须明确其
诊断监督用途，不能把它改名成无监督检测器。目前没有训练任何GNN或新分类器。

## 与最接近已有工作的关系

[CHARM v2](https://arxiv.org/html/2509.24770)已经将token激活和跨层/头attention
表示成属性图，并监督训练GNN；其零样本是跨数据集迁移，不是无标签学习。
其默认实现删除prompt内部边并阈值稀疏化，本次为归属机制检查保留这些信息。
这种表示本身不是新颖性结论；本项目待证的是约束归属及错误决策的可辨与采纳机制。
一手论文与代码已人工核查；自动verify_papers助手不可用，程序状态UNVERIFIED。

## 本地执行

从reanchor运行（已存在的本地环境，不安装或下载）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=../graph:src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m decoding.sample_graph_capture --output outputs/attributed_samples_20260912
```

目录必须不存在。完成标记是根目录manifest.json，settings与执行源码先保存，
两个样本全部捕获/存档后才写summary和manifest。未出现完成标记的目录不得当完整结果。

## 实际执行记录

独立代理按上述命令原样执行一次，退出0。00012/00013分别730/681节点，全部
32层、32头、4096维；数组合计6,509,746,107字节，存档约6.1GiB。runner记录
14.93秒（不含解释器启动与最终哈希计算），GPU峰值约16.54GiB。

最大attention行和误差0.003023/0.002947，符合bf16概率舍入；没有人工归一化改写。
raw残差相加的最大float32差值均0.75，单此绝对值不能判定异常或正确：原计算
在bf16中相加。独立witness检查两个实际图的所有层，按bf16舍入后残差相加
**零不一致元素**；33项文件哈希全部相符。结论记录在
refine-logs/sample_graph_witness_20260912.md。validator本身仅报告原始相加误差，
未自动强制dtype相关门槛，工程审查将加强此检查列为可选改进。

原始图在reanchor/outputs/attributed_samples_20260912，完整manifest包含源码和
数组哈希；只保留一份大数组，未复制进小型结果包。新增2个实际tiny Llama测试通过。
独立工程审查见refine-logs/sample_graph_engineering_review_20260912.md。
图包含归属信息的程度、联合拓扑增益、自动回看定位率仍未评价。
